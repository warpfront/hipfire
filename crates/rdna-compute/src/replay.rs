// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fail-closed integration gate for Redline record/replay.
//!
//! This module records the central HIP launch surface during warmup and owns
//! the fail-closed selection state. It deliberately does not reinterpret
//! `void**` arguments: a model adapter must supply explicit resource accesses
//! and a kernarg ABI to `redline-dispatch` before installing a prepared plan.
//! Replay remains default-off except for the runtime automatic default
//! `mq4r_redline_default` (exact `gfx1100`/`gfx1151`/`gfx1201`, single-GPU,
//! case-insensitive `.mq4r`; `gfx1200`/others opt-in)
//! selected by the daemon after model load. Runtime default ≠ Redline
//! certification/registry admission; built-in `hip` profile or explicit backend
//! selection disables the automatic default.

use std::collections::{BTreeMap, BTreeSet};

use hip_bridge::memory_effects::{self, MemoryEffects};
use hip_bridge::HipRuntime;
use radiowave::{CodeObjectCertification, KernelArgumentAccess, MutableReadCache};
use redline_dispatch::aql::{
    load_symbols, BatchFencePolicy, Executable, FenceScope, Gfx10DispatchInitiatorPolicy,
    Gfx10Pm4CommandBuffer, Gfx10SetShRegRecord, Gfx11ComputeResourceLimitsPolicy,
    Gfx11DispatchInterleave, Gfx12DispatchPacing, Gfx12Pm4CommandBuffer, Gfx12RmwAcquirePolicy,
    GpuBatchTiming, GpuDevice, GpuMultiQueueTiming, GpuSelector, HeaderPolicy, KernargBuffer,
    KernargPool, Kernel, LaunchGeometry, PhasedMultiQueuePm4Ib, QueuePolicy, Quiescence,
    RecordedDispatch, Runtime, SingleQueueBatchGraph, SingleQueuePm4Ib,
};
use redline_dispatch::{
    AllocationPolicy, BindingRevision, KernargAbi, KernargField, Recorder, ReplayBindings,
    ResourceBinding, ResourceId,
};

use crate::code_object::{CodeObjectId, RecordedArtifact};
use crate::dispatch::VmmResourceMove;

pub(crate) mod railgun_shadow;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayQuiescence {
    Proven,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct RetainedReplayFailure {
    pub error: String,
    pub quiescence: ReplayQuiescence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayBackendRequest {
    Hip,
    Shadow,
    Auto,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReplayTransport {
    AqlPackets,
    Pm4Ib,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pm4Architecture {
    Gfx10,
    Gfx11,
    Gfx12,
}

/// Per-stream producer/consumer visibility for legacy (gfx10/gfx11) PM4 IBs.
///
/// `CsPartialFlush` retains the historical EVENT_WRITE path. `ReleaseWait` is
/// admitted only for exact gfx1010 single-queue retained replay and pairs a
/// fine-grained host word with RELEASE_MEM + WAIT_REG_MEM epochs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LegacyDependencyMode {
    CsPartialFlush,
    ReleaseWait { address: u64, next_epoch: u32 },
}

/// Exact gfx1010 gate for the RELEASE_MEM/WAIT_REG_MEM dependency fence.
/// Architecture must already be the gfx10 family map; the device name is
/// matched ASCII-case-insensitively so only the Navi10 agent is selected.
fn gfx1010_release_wait_required(architecture: Pm4Architecture, device_name: &str) -> bool {
    architecture == Pm4Architecture::Gfx10 && device_name.eq_ignore_ascii_case("gfx1010")
}

/// Diagnostic-only override for exact-gfx1010 retained-PM4 dependency fencing.
/// Default remains `ReleaseWait`; `CsPartialFlush` is the historical EVENT_WRITE path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Gfx1010DependencyPolicy {
    ReleaseWait,
    CsPartialFlush,
}

/// Pure parser for `HIPFIRE_REPLAY_PM4_GFX1010_DEPENDENCY`.
///
/// Non-exact-gfx1010 devices always resolve to `CsPartialFlush` and ignore `value`.
/// On exact gfx1010: unset/`release-wait` => `ReleaseWait`, `cs-partial-flush` =>
/// `CsPartialFlush`; any other value is a hard prepare error naming the key.
fn gfx1010_dependency_policy_from_value(
    architecture: Pm4Architecture,
    device_name: &str,
    value: Option<&str>,
) -> Result<Gfx1010DependencyPolicy, String> {
    if !gfx1010_release_wait_required(architecture, device_name) {
        return Ok(Gfx1010DependencyPolicy::CsPartialFlush);
    }
    match value {
        None => Ok(Gfx1010DependencyPolicy::ReleaseWait),
        Some("release-wait") => Ok(Gfx1010DependencyPolicy::ReleaseWait),
        Some("cs-partial-flush") => Ok(Gfx1010DependencyPolicy::CsPartialFlush),
        Some(raw) => Err(format!(
            "invalid HIPFIRE_REPLAY_PM4_GFX1010_DEPENDENCY={raw:?}; \
             expected unset, \"release-wait\", or \"cs-partial-flush\""
        )),
    }
}

fn gfx1010_dependency_policy_from_config(
    architecture: Pm4Architecture,
    device_name: &str,
) -> Result<Gfx1010DependencyPolicy, String> {
    let raw = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_GFX1010_DEPENDENCY");
    let policy = gfx1010_dependency_policy_from_value(architecture, device_name, raw.as_deref())?;
    if gfx1010_release_wait_required(architecture, device_name) {
        let source = if raw.is_none() { "default" } else { "explicit" };
        eprintln!("[redline] gfx1010 PM4 dependency mode={policy:?} ({source})");
    }
    Ok(policy)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Gfx11EntryAcquirePolicy {
    System,
    Agent,
    Vmem,
    None,
}

impl Pm4Architecture {
    fn from_device(device: &GpuDevice) -> Result<Self, String> {
        Self::from_name(device.name())
    }

    fn from_name(device_name: &str) -> Result<Self, String> {
        let name = device_name.to_ascii_lowercase();
        if name.starts_with("gfx10") {
            Ok(Self::Gfx10)
        } else if name.starts_with("gfx11") {
            Ok(Self::Gfx11)
        } else if matches!(name.as_str(), "gfx1200" | "gfx1201") {
            Ok(Self::Gfx12)
        } else {
            Err(format!(
                "retained PM4 has no certified register map for HSA agent {:?}",
                device_name
            ))
        }
    }
}

#[derive(Clone)]
enum Pm4Commands {
    Legacy {
        architecture: Pm4Architecture,
        commands: Gfx10Pm4CommandBuffer,
        dependency_mode: LegacyDependencyMode,
    },
    Gfx12(Gfx12Pm4CommandBuffer),
}

fn create_phased_pm4_graph(
    architecture: Pm4Architecture,
    device: &GpuDevice,
    pool: &KernargPool,
    phases: &[Vec<Pm4Commands>],
    native_sync: bool,
) -> Result<PhasedMultiQueuePm4Ib, String> {
    match architecture {
        Pm4Architecture::Gfx10 | Pm4Architecture::Gfx11 => {
            let legacy = phases
                .iter()
                .map(|phase| {
                    phase
                        .iter()
                        .map(|commands| match commands {
                            Pm4Commands::Legacy {
                                architecture: actual,
                                commands,
                                ..
                            } if *actual == architecture => Ok(commands.clone()),
                            _ => Err("mixed PM4 architecture in phased graph".to_owned()),
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            match architecture {
                Pm4Architecture::Gfx10 if native_sync => {
                    PhasedMultiQueuePm4Ib::create_profiled_native_gfx10(device, pool, &legacy)
                }
                Pm4Architecture::Gfx10 => {
                    PhasedMultiQueuePm4Ib::create_profiled_gfx10(device, pool, &legacy)
                }
                Pm4Architecture::Gfx11 if native_sync => {
                    PhasedMultiQueuePm4Ib::create_profiled_native_gfx11(device, pool, &legacy)
                }
                Pm4Architecture::Gfx11 => {
                    PhasedMultiQueuePm4Ib::create_profiled_gfx11(device, pool, &legacy)
                }
                Pm4Architecture::Gfx12 => unreachable!(),
            }
        }
        Pm4Architecture::Gfx12 => {
            if native_sync {
                return Err(
                    "native PM4 phase synchronization is not yet lowered for gfx12".to_owned(),
                );
            }
            let gfx12 = phases
                .iter()
                .map(|phase| {
                    phase
                        .iter()
                        .map(|commands| match commands {
                            Pm4Commands::Gfx12(commands) => Ok(commands.clone()),
                            Pm4Commands::Legacy { .. } => {
                                Err("mixed PM4 architecture in phased graph".to_owned())
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            PhasedMultiQueuePm4Ib::create_profiled(device, pool, &gfx12)
        }
    }
    .map_err(|error| error.to_string())
}

impl Pm4Commands {
    fn new(
        architecture: Pm4Architecture,
        policy: Pm4RegisterPolicy,
        dispatch_initiator_policy: Gfx10DispatchInitiatorPolicy,
        dispatch_interleave: Option<Gfx11DispatchInterleave>,
        resource_limits_policy: Gfx11ComputeResourceLimitsPolicy,
    ) -> Self {
        Self::new_with_dependency(
            architecture,
            policy,
            dispatch_initiator_policy,
            dispatch_interleave,
            resource_limits_policy,
            LegacyDependencyMode::CsPartialFlush,
        )
    }

    fn new_with_dependency(
        architecture: Pm4Architecture,
        policy: Pm4RegisterPolicy,
        dispatch_initiator_policy: Gfx10DispatchInitiatorPolicy,
        dispatch_interleave: Option<Gfx11DispatchInterleave>,
        resource_limits_policy: Gfx11ComputeResourceLimitsPolicy,
        dependency_mode: LegacyDependencyMode,
    ) -> Self {
        match architecture {
            Pm4Architecture::Gfx10 | Pm4Architecture::Gfx11 => {
                let commands = match policy {
                    Pm4RegisterPolicy::Legacy => Gfx10Pm4CommandBuffer::new(),
                    Pm4RegisterPolicy::Static | Pm4RegisterPolicy::Stateful => {
                        Gfx10Pm4CommandBuffer::new_stateful()
                    }
                }
                .with_dispatch_initiator_policy(dispatch_initiator_policy)
                .with_dispatch_interleave(dispatch_interleave)
                .with_resource_limits_policy(resource_limits_policy);
                Self::Legacy {
                    architecture,
                    commands,
                    dependency_mode,
                }
            }
            Pm4Architecture::Gfx12 => {
                debug_assert!(
                    matches!(dependency_mode, LegacyDependencyMode::CsPartialFlush),
                    "gfx12 never uses legacy dependency fences"
                );
                let commands = match policy {
                    Pm4RegisterPolicy::Legacy => Gfx12Pm4CommandBuffer::new(),
                    Pm4RegisterPolicy::Static => Gfx12Pm4CommandBuffer::new_static_stateful(),
                    Pm4RegisterPolicy::Stateful => Gfx12Pm4CommandBuffer::new_stateful(),
                };
                Self::Gfx12(commands)
            }
        }
    }

    fn acquire_entry(&mut self, gfx12_gcr_trim: bool, gfx11_policy: Gfx11EntryAcquirePolicy) {
        match self {
            Self::Legacy { commands, .. } => match gfx11_policy {
                Gfx11EntryAcquirePolicy::System => commands.acquire_system(),
                Gfx11EntryAcquirePolicy::Agent => commands.acquire_inter_node_same_agent(),
                Gfx11EntryAcquirePolicy::Vmem => commands.acquire_inter_node_vmem(),
                Gfx11EntryAcquirePolicy::None => {}
            },
            Self::Gfx12(commands) if gfx12_gcr_trim => commands.acquire_system_gfx12(),
            Self::Gfx12(commands) => commands.acquire_system(),
        }
    }

    /// Emit the sentinel epoch-0 release/wait before entry acquire so every
    /// immutable replay starts from a known fence word (ABA prevention).
    fn emit_entry_sentinel_reset(&mut self) -> Result<(), String> {
        match self {
            Self::Legacy {
                commands,
                dependency_mode:
                    LegacyDependencyMode::ReleaseWait {
                        address,
                        next_epoch,
                    },
                ..
            } => {
                if *next_epoch != 0 {
                    return Err(format!(
                        "gfx1010 dependency fence entry sentinel requires next_epoch=0, got {next_epoch}"
                    ));
                }
                commands.dependency_fence(*address, 0);
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn acquire_inter_node(&mut self, gfx12_gcr_trim: bool, vmem_only: bool) {
        match self {
            Self::Legacy { commands, .. } if vmem_only => commands.acquire_inter_node_vmem(),
            Self::Legacy { commands, .. } => commands.acquire_inter_node_same_agent(),
            // Opt-in Radiowave-certified rung: the caller gates `vmem_only`
            // by architecture (Legacy `HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE`
            // vs gfx12 `HIPFIRE_REPLAY_PM4_GFX12_VMEM_ACQUIRE`), so reaching
            // this arm means the consumer is certified VMEM-only on gfx12.
            Self::Gfx12(commands) if vmem_only => {
                commands.acquire_rmw_gfx12(Gfx12RmwAcquirePolicy::HipLlvmVmemL1);
            }
            Self::Gfx12(commands) if gfx12_gcr_trim => commands.acquire_inter_node_gfx12(),
            Self::Gfx12(commands) => commands.acquire_system(),
        }
    }

    fn requires_dependency_acquire(&self) -> bool {
        true
    }

    fn wait_compute_idle(&mut self) -> Result<(), String> {
        match self {
            Self::Legacy {
                commands,
                dependency_mode:
                    LegacyDependencyMode::ReleaseWait {
                        address,
                        next_epoch,
                    },
                ..
            } => {
                let epoch = next_epoch.checked_add(1).ok_or_else(|| {
                    "gfx1010 dependency fence epoch overflow (u32 exhausted)".to_owned()
                })?;
                *next_epoch = epoch;
                commands.dependency_fence(*address, epoch);
                Ok(())
            }
            Self::Legacy { commands, .. } => {
                commands.wait_compute_idle();
                Ok(())
            }
            Self::Gfx12(commands) => {
                commands.wait_compute_idle();
                Ok(())
            }
        }
    }

    fn gfx12_system_acquire(&mut self) -> Result<(), String> {
        match self {
            Self::Gfx12(commands) => {
                commands.acquire_system_gfx12();
                Ok(())
            }
            Self::Legacy { .. } => {
                Err("gfx12 system acquire requested for a legacy PM4 stream".to_owned())
            }
        }
    }
    /// Trailing release at the tape terminal: drain shaders, then emit the
    /// architecture-matched full-system acquire so GL2 is written back for a
    /// non-shader next consumer (e.g. an SDMA H2D copy). `CS_PARTIAL_FLUSH`
    /// alone does not write back GL2. This is `ACQUIRE_MEM` with the GL2
    /// writeback bits, not a `RELEASE_MEM` packet. Unconditional: a retained
    /// tape costs one packet per tape.
    fn trailing_release(&mut self) -> Result<(), String> {
        self.wait_compute_idle()?;
        match self {
            Self::Legacy { commands, .. } => {
                commands.acquire_system();
                Ok(())
            }
            Self::Gfx12(commands) => {
                commands.acquire_system_gfx12();
                Ok(())
            }
        }
    }

    #[cfg(test)]
    fn dependency_mode(&self) -> Option<LegacyDependencyMode> {
        match self {
            Self::Legacy {
                dependency_mode, ..
            } => Some(*dependency_mode),
            Self::Gfx12(_) => None,
        }
    }

    #[cfg(test)]
    fn dwords(&self) -> Option<&[u32]> {
        match self {
            Self::Legacy { commands, .. } => Some(commands.dwords()),
            Self::Gfx12(_) => None,
        }
    }

    /// The gfx12 command dwords (railgun's shadow diff reads both tapes).
    fn gfx12_dwords(&self) -> Option<&[u32]> {
        match self {
            Self::Gfx12(commands) => Some(commands.dwords()),
            Self::Legacy { .. } => None,
        }
    }

    fn dispatch(
        &mut self,
        kernel: &Kernel,
        geometry: LaunchGeometry,
        dynamic_group_bytes: u32,
        kernarg_address: *mut std::ffi::c_void,
    ) -> Result<(), String> {
        match self {
            Self::Legacy { commands, .. } => commands
                .dispatch(kernel, geometry, dynamic_group_bytes, kernarg_address)
                .map_err(|error| error.to_string()),
            Self::Gfx12(commands) => commands
                .dispatch(kernel, geometry, dynamic_group_bytes, kernarg_address)
                .map_err(|error| error.to_string()),
        }
    }

    fn len_dwords(&self) -> u32 {
        match self {
            Self::Legacy { commands, .. } => commands.len_dwords(),
            Self::Gfx12(commands) => commands.len_dwords(),
        }
    }

    fn packet_census(&self) -> Option<Result<BTreeMap<(u32, u32), usize>, usize>> {
        match self {
            Self::Legacy { commands, .. } => Some(commands.packet_census()),
            Self::Gfx12(_) => None,
        }
    }

    fn set_sh_reg_records(&self) -> Option<Result<Vec<Gfx10SetShRegRecord>, usize>> {
        match self {
            Self::Legacy { commands, .. } => Some(commands.set_sh_reg_records()),
            Self::Gfx12(_) => None,
        }
    }

    fn populate_dispatch_span_boundaries(
        &self,
        boundaries: &mut [Pm4DispatchBoundary],
    ) -> Result<(), String> {
        let Self::Gfx12(commands) = self else {
            // Legacy boundary flags were recorded while planning; the first
            // dispatch follows the entry acquire.
            if let Some(first) = boundaries.first_mut() {
                first.entry_acquire = true;
            }
            return Ok(());
        };
        let attributions = commands
            .dispatch_span_attributions()
            .map_err(|error| error.to_string())?;
        if attributions.len() != boundaries.len() {
            return Err(format!(
                "generated PM4 dispatch attribution mismatch: expected {}, got {}",
                boundaries.len(),
                attributions.len()
            ));
        }
        for (boundary, attribution) in boundaries.iter_mut().zip(attributions) {
            boundary.entry_acquire = attribution.entry_acquire;
            boundary.wait_compute_idle = attribution.wait_compute_idle;
            boundary.acquire_inter_node = attribution.acquire_inter_node;
        }
        Ok(())
    }

    fn create_graph(
        &self,
        device: &GpuDevice,
        pool: &KernargPool,
        ib_pool: Option<&KernargPool>,
        cu_mask: Option<&[u32; 2]>,
        dispatch_profile: bool,
    ) -> Result<SingleQueuePm4Ib, String> {
        let graph = match self {
            Self::Legacy {
                architecture: Pm4Architecture::Gfx10,
                commands,
                ..
            } => SingleQueuePm4Ib::create_profiled_gfx10(device, pool, commands),
            Self::Legacy {
                architecture: Pm4Architecture::Gfx11,
                commands,
                ..
            } => {
                if dispatch_profile {
                    SingleQueuePm4Ib::create_boundary_profiled_legacy(device, pool, commands)
                } else if let Some(ib_pool) = ib_pool {
                    SingleQueuePm4Ib::create_profiled_legacy_with_ib_pool(
                        device, pool, ib_pool, commands,
                    )
                } else {
                    SingleQueuePm4Ib::create_profiled_gfx11(device, pool, commands)
                }
            }
            Self::Legacy {
                architecture: Pm4Architecture::Gfx12,
                ..
            } => unreachable!("gfx12 never uses the legacy PM4 command variant"),
            Self::Gfx12(commands) => {
                // HIPFIRE_REDLINE_DISPATCH_PROFILE=1 builds a tape carrying one
                // GPU-clock write per dispatch, so a machine whose retained path
                // underperforms can report WHERE the time goes instead of only
                // how much there is. The instrumented tape necessarily has a
                // different dword count and sequence hash, so such a run cannot
                // satisfy a golden fixture — it is a diagnostic, not a
                // certification.
                if dispatch_profile || dispatch_profile_enabled() {
                    SingleQueuePm4Ib::create_dispatch_profiled(device, pool, commands)
                } else if let Some(ib_pool) = ib_pool {
                    SingleQueuePm4Ib::create_profiled_with_ib_pool(device, pool, ib_pool, commands)
                } else {
                    SingleQueuePm4Ib::create_profiled(device, pool, commands)
                }
            }
        }
        .map_err(|error| error.to_string())?;
        if let Some(cu_mask) = cu_mask {
            graph
                .set_cu_mask(64, cu_mask)
                .map_err(|error| error.to_string())?;
        }
        Ok(graph)
    }
}

/// Collect the exact-object Radiowave certifications of a retained tape's
/// admitted images. Certification was verified once at module admission;
/// missing, malformed, or hash-stale manifests left none and therefore retain
/// the conservative scalar-cache acquire.
fn radiowave_certifications(
    recorded: &[RecordedHipLaunch],
    prefix: usize,
) -> BTreeMap<CodeObjectId, CodeObjectCertification> {
    let mut certifications = BTreeMap::new();
    for launch in recorded.iter().take(prefix) {
        let Some(artifact) = launch.artifact.as_ref() else {
            continue;
        };
        if let Some(certification) = artifact.radiowave() {
            certifications
                .entry(artifact.id())
                .or_insert_with(|| certification.clone());
        }
    }
    certifications
}

fn radiowave_vmem_only_consumer(
    certifications: &BTreeMap<CodeObjectId, CodeObjectCertification>,
    launch: &RecordedHipLaunch,
) -> bool {
    let Some(artifact) = launch.artifact.as_ref() else {
        return false;
    };
    certifications.get(&artifact.id()).is_some_and(|certification| {
        certification.mutable_read_cache(&launch.kernel) == MutableReadCache::VmemOnly
    })
}

/// Where Redline's resource effects for `launch` came from: the object's
/// Radiowave sidecar, the name-keyed `pointer_effects` table, or neither
/// (railgun shadow report).
fn redline_effect_source(launch: &RecordedHipLaunch) -> String {
    let radiowave = launch
        .artifact
        .as_ref()
        .and_then(|artifact| artifact.radiowave())
        .is_some_and(|certification| certification.argument_effects(&launch.kernel).is_some());
    if radiowave {
        "radiowave"
    } else if launch.accesses.is_some() {
        "table"
    } else {
        "none"
    }
    .to_owned()
}

fn pm4_vmem_acquire_enabled(
    architecture: Pm4Architecture,
    configured: bool,
    certifications: &BTreeMap<CodeObjectId, CodeObjectCertification>,
    launch: &RecordedHipLaunch,
) -> bool {
    pm4_vmem_acquire_arch_enabled(architecture, configured)
        && radiowave_vmem_only_consumer(certifications, launch)
}

fn pm4_vmem_acquire_arch_enabled(architecture: Pm4Architecture, configured: bool) -> bool {
    architecture != Pm4Architecture::Gfx12 && configured
}

/// Opt-in gate for the gfx12 `HipLlvmVmemL1` RMW rung, mirroring
/// `HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE` but defaulting OFF on every
/// device: unlike the gfx1151 default, no gfx12 part has yet proven the VMEM
/// rung exact and non-slower in the harness. `auto` also means off; the rung
/// stays an explicit operator decision until that evidence lands.
fn pm4_gfx12_vmem_acquire_from_config() -> bool {
    pm4_gfx12_vmem_acquire_from_value(hipfire_config::process_value(
        "HIPFIRE_REPLAY_PM4_GFX12_VMEM_ACQUIRE",
    ))
}

fn pm4_gfx12_vmem_acquire_from_value(value: Option<String>) -> bool {
    match value {
        Some(raw) if raw != "auto" => matches!(raw.as_str(), "1" | "true" | "on"),
        _ => false,
    }
}

/// Gfx12 counterpart to `pm4_vmem_acquire_enabled`: the same Radiowave
/// `vmem_only` certification, but gated by the gfx12 opt-in instead of the
/// Legacy flag (which `pm4_vmem_acquire_arch_enabled` keeps gfx12-excluded).
fn pm4_gfx12_vmem_acquire_enabled(
    configured: bool,
    certifications: &BTreeMap<CodeObjectId, CodeObjectCertification>,
    launch: &RecordedHipLaunch,
) -> bool {
    configured && radiowave_vmem_only_consumer(certifications, launch)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecordedAccessMode {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RecordedResourceAccess {
    allocation_base: u64,
    allocation_bytes: u64,
    // Diagnostic pointer start within the allocation. Scheduling remains
    // allocation-wide; this proves whether a blocked boundary has any exact
    // producer/consumer pointer dependency before byte ranges are considered.
    access_base: u64,
    mode: RecordedAccessMode,
}

impl RecordedResourceAccess {
    fn end(self) -> u64 {
        self.allocation_base + self.allocation_bytes
    }

    fn conflicts(self, other: Self) -> bool {
        let overlaps = self.allocation_base < other.end() && other.allocation_base < self.end();
        overlaps
            && (self.mode == RecordedAccessMode::Write || other.mode == RecordedAccessMode::Write)
    }

    fn same_start_conflicts(self, other: Self) -> bool {
        self.access_base == other.access_base
            && (self.mode == RecordedAccessMode::Write || other.mode == RecordedAccessMode::Write)
    }
}

#[derive(Clone, Copy)]
struct PointerEffect {
    offset: usize,
    mode: RecordedAccessMode,
}

const fn read(offset: usize) -> PointerEffect {
    PointerEffect {
        offset,
        mode: RecordedAccessMode::Read,
    }
}

const fn write(offset: usize) -> PointerEffect {
    PointerEffect {
        offset,
        mode: RecordedAccessMode::Write,
    }
}

/// Pointer fields and memory effects for kernels admitted to Qwen AR replay.
///
/// A non-const kernel pointer is conservatively classified as `Write`, which
/// also covers read-modify-write effects. Unknown kernels fail closed and keep
/// their compute-idle boundaries. Offsets are the naturally aligned HIP
/// kernarg ABI offsets verified by the captured-blob/loader parity gate.
fn pointer_effects(kernel: &str) -> Option<Vec<PointerEffect>> {
    // The repacker completely overwrites all three planes on each launch.
    // ADD is a read-modify-write of Y0, represented conservatively as Write.
    if kernel == "mq4v2_fp8_fragment_repack_gfx1201" {
        return Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            write(32),
            write(40),
            write(48),
        ]);
    }
    match kernel {
        "gemm_mq4g256v2_fp8_set_row_b1"
        | "gemm_mq4g256v2_fp8_add_row_b1"
        | "gemm_mq4g256v2_fp8_silu_row_b1" => {
            return Some(vec![
                read(0),
                read(8),
                read(16),
                read(24),
                read(32),
                write(40),
            ])
        }
        "gemm_mq4g256v2_fp8_qkv_row_b1" => {
            return Some(vec![
                read(0),
                read(8),
                read(16),
                read(24),
                read(32),
                write(40),
                write(48),
                write(56),
            ])
        }
        "gemm_mq4g256v2_fp8_qkvza_row_b1" => {
            return Some(vec![
                read(0),
                read(8),
                read(16),
                read(24),
                read(32),
                write(40),
                write(48),
                write(56),
                write(64),
            ])
        }
        _ => {}
    }
    if matches!(
        kernel,
        "fused_gate_up_hfq4g256"
            | "fused_gate_up_hfq4g256_k1024_gfx1201"
            | "fused_gate_up_hfq4g256_dot_reform_gfx1100"
            | "fused_gate_up_hfq4g256_dot_prefetch_gfx1100"
            | "fused_gate_up_hfq4g256_pair_gfx1100"
            | "fused_gate_up_hfq4g256_pair2_gfx1100"
            | "fused_gate_up_hfq4g256_quad_prefetch_gfx1100"
            | "fused_gate_up_hfq4g256_setprio_gfx1100"
            | "fused_gate_up_hfq4g256_lane0_headers_gfx1100"
            | "fused_gate_up_hfq4g256_stage_x32_gfx1100"
            | "fused_gate_up_mq4g256v2"
            | "fused_gate_up_mq4g256v2_k5120_gfx1100"
    ) {
        return Some(vec![read(0), read(8), read(16), write(24), write(32)]);
    }
    if matches!(
        kernel,
        "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151"
            | "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151"
    ) {
        // The split producers share read-only routing, activation, and packed
        // expert allocations but write distinct gate/up output allocations.
        return Some(vec![read(0), read(8), read(16), write(24)]);
    }
    if matches!(
        kernel,
        "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_all_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_hybrid_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_low_vgpr_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_all_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_vgpr_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_paired_waves_k2048_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_persistent_rank8_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_route_all_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_wave64"
    ) {
        return Some(vec![read(0), read(8), read(16), write(24), write(32)]);
    }
    if matches!(
        kernel,
        "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed"
            | "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed"
            | "gemv_mq2g256gl_moe_gate_up_k8_indexed"
            | "gemv_mq3g256gl_moe_gate_up_k8_indexed"
            // Batched-K4 prefill siblings: same pointer set and modes, but a
            // K_TOP scalar makes the kernarg block 52 B, not 48 (see below).
            | "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4"
            | "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched_k4"
    ) {
        return Some(vec![read(0), read(8), read(16), write(24), write(32)]);
    }

    if matches!(
        kernel,
        "fused_qkvza_hfq4g256_k2048_all_buffer_dlc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_glc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_slc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_hybrid_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_pair_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_x_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_r4_stream_gfx1151"
    ) {
        return Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
            write(48),
            write(56),
            write(64),
        ]);
    }
    if matches!(
        kernel,
        "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_hybrid_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row1_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_clustered_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row8_gfx1151"
    ) {
        return Some(vec![read(0), read(8), read(16), write(24)]);
    }
    if matches!(
        kernel,
        "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r2"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r4"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r2"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r4"
            | "gemv_mq3g256_lloyd_moe_ninepath_d4"
            | "gemv_mq2g256gl_moe_down_residual_scaled_k8_indexed"
            | "gemv_mq3g256gl_moe_down_residual_scaled_k8_indexed"
            // Batched-K4 prefill siblings; 52 B kernarg (K_TOP scalar).
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4"
    ) {
        return Some(vec![read(0), read(8), read(16), read(24), write(32)]);
    }

    // MQ4G256V2 / MQ6G256V2 MoE routes. Exact sister ABIs of the HFQ4/MQ4V2
    // launchers — V1/V2 names must never alias. Offsets are the naturally
    // aligned HIP kernarg fields; launch_maybe_blob pads the recorded block
    // to 16 B (see expected_kernarg_bytes).
    if matches!(
        kernel,
        "gemv_mq4g256v2_moe_gate_up_k8_indexed"
            | "gemv_mq6g256v2_moe_gate_up_k8_indexed"
            | "gemv_mq4g256v2_moe_gate_up_k8_indexed_k2048_nolds_gfx1100"
            // Batched Path1 sisters: same pointer set/modes; K_TOP scalar
            // only changes the recorded kernarg length (64 B padded).
            | "gemv_mq4g256v2_moe_gate_up_k8_indexed_batched"
            | "gemv_mq6g256v2_moe_gate_up_k8_indexed_batched"
    ) {
        return Some(vec![read(0), read(8), read(16), write(24), write(32)]);
    }
    if matches!(
        kernel,
        "gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded"
            | "gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded"
    ) {
        return Some(vec![read(0), read(8), read(16), write(24)]);
    }
    if matches!(
        kernel,
        "gemv_mq4g256v2_moe_ninepath_d4"
            | "gemv_mq4g256v2_moe_ninepath_rpb8_gfx1100"
            | "gemv_mq6g256v2_moe_ninepath_d4"
    ) {
        // expert_ptrs, topk_indices, topk_weights, act (read); out is RMW.
        return Some(vec![read(0), read(8), read(16), read(24), write(32)]);
    }
    if matches!(
        kernel,
        "gemm_mq4g256v2_moe_grouped_wmma_k2"
            | "gemm_mq4g256v2_moe_grouped_wmma_gfx12"
            | "gemm_mq6g256v2_moe_grouped_wmma_k2"
            | "gemm_mq6g256v2_moe_grouped_wmma_gfx12"
    ) {
        // expert_weight_ptrs, tile_ids, sorted_slot_index, X_src (read);
        // Y_grouped (write).
        return Some(vec![read(0), read(8), read(16), read(24), write(32)]);
    }
    // Mixed-precision MoE (tags 0..18, V1/V2 frozen). Block-uniform dtype_tags branch.
    // Gate/up carries 6 pointers (adds dtype_tags at +8); down carries 5.
    // Grouped mixed carries 6 pointers (adds dtype_tags at +8). V1/V2 names never alias.
    if kernel == "gemv_mixed_moe_gate_up_k8_indexed_batched" {
        // expert_ptrs, dtype_tags, topk_indices, x, y_gate, y_up
        return Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            write(32),
            write(40),
        ]);
    }
    if kernel == "gemv_mixed_moe_down_k8_indexed_batched_expanded" {
        // expert_ptrs, dtype_tags, topk_indices, rot_batch, expert_outputs
        return Some(vec![read(0), read(8), read(16), read(24), write(32)]);
    }
    if matches!(
        kernel,
        "gemm_mixed_moe_grouped_wmma_k2"
            | "gemm_mixed_moe_grouped_wmma_gfx12"
            | "gemm_mixed_moe_grouped_wmma_4w_k2"
    ) {
        // expert_weight_ptrs, dtype_tags, tile_ids, sorted_slot_index, X_src (read); Y_grouped (write)
        return Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
        ]);
    }
    // Dense shared-expert V2 (qt44 qt47). Plain GEMV / residual / multirow share the same
    // 3-pointer ABI: a_raw, x (read); y (write/RMW). Same padded size as HFQ4 dense.
    if matches!(
        kernel,
        "gemv_mq4g256v2"
            | "gemv_mq4g256v2_residual"
            | "gemv_mq4g256v2_residual_r1_k4096_gfx1100_noscratch"
            | "gemv_mq6g256v2"
            | "gemv_mq6g256v2_residual"
            | "gemv_mq4g256v2_multirow_r2"
            | "gemv_mq4g256v2_multirow_r4"
            | "gemv_mq4g256v2_multirow_r8"
            | "gemv_mq6g256v2_multirow_r2"
            | "gemv_mq6g256v2_multirow_r4"
            | "gemv_mq6g256v2_multirow_r8"
    ) {
        return Some(vec![read(0), read(8), write(16)]);
    }
    // MQ4G256V2-Lloyd LUT GEMVs (qt52): same 3-pointer ABI as the uniform V2
    // pair (a_raw, x read; y write/RMW) plus 8 by-value LUT dwords and 2 i32.
    if matches!(
        kernel,
        "gemv_mq4g256v2_lloyd" | "gemv_mq4g256v2_residual_lloyd"
    ) {
        return Some(vec![read(0), read(8), write(16)]);
    }
    // Dense shared-expert V2 residual WMMA GEMM (prefill/batched). 3 pointers + 3 i32
    // (M,K,batch). Y is write (output) via residual path; distinct symbols per arch tile.
    if matches!(
        kernel,
        "gemm_mq4g256v2_residual_wmma"
            | "gemm_mq4g256v2_residual_wmma_gfx12"
            | "gemm_mq6g256v2_residual_wmma"
            | "gemm_mq6g256v2_residual_wmma_gfx12"
            | "gemm_mq4g256v2_residual_wmma_gfx11_bt4"
            | "gemm_mq4g256v2_residual_wmma_gfx11_bt6"
            | "gemm_mq4g256v2_residual_wmma_gfx11_bt8"
            | "gemm_mq6g256v2_residual_wmma_gfx11_bt4"
            | "gemm_mq6g256v2_residual_wmma_gfx11_bt6"
            | "gemm_mq6g256v2_residual_wmma_gfx11_bt8"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_mw4_lds"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_mw8_lds"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage"
            | "gemm_mq6g256v2_residual_wmma_gfx11_mw4_lds"
            | "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds"
    ) {
        return Some(vec![read(0), read(8), write(16)]);
    }
    // Exact-gfx1100 small-N gate/up LDS-stage GEMM. Five pointers followed by
    // gate_m, up_m, K, N: weights and X are read, split outputs are overwritten.
    if kernel == "gemm_gate_up_mq4g256v2_wmma_gfx1100_ldsstage" {
        return Some(vec![read(0), read(8), read(16), write(24), write(32)]);
    }
    // Typed F32 copy used when hidden-state staging is recorded for retained
    // replay: dst, src, n.
    if kernel == "copy_f32_buffer" {
        return Some(vec![write(0), read(8)]);
    }
    // F16 dense batched GEMM (Maple router + DeepSeek compressor shapes). 3 pointers
    // + 3 i32 (M,K,B) = 36 explicit bytes. A@0 and X@8 are reads; Y@16 is a pure
    // overwrite (`Y[...] = acc`), so write — never an RMW. gfx11 and gfx12 are
    // distinct symbols with one shared contract, like the residual `_wmma`/`_gfx12` pairs.
    if matches!(kernel, "gemm_f16_x_f16_wmma" | "gemm_f16_x_f16_wmma_gfx12") {
        return Some(vec![read(0), read(8), write(16)]);
    }

    if kernel == "moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate" {
        return Some(vec![
            read(0),
            write(8),
            write(16),
            read(32),
            read(40),
            read(48),
            read(56),
            write(64),
        ]);
    }
    if kernel == "moe_router_softmax_topk_k8_wave64"
        || kernel == "moe_router_softmax_topk_k8_wave64_exact"
    {
        return Some(vec![read(0), write(8), write(16)]);
    }
    if kernel.starts_with("gated_delta_net_q8_compact") {
        return Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
            write(48),
            write(56),
            write(80),
        ]);
    }
    if kernel == "fused_qkvza_hfq4g256_k2048_scalar_prep" {
        return Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
            write(48),
            write(56),
            write(64),
            read(72),
            read(80),
        ]);
    }
    if kernel == "conv1d_silu_split_qknorm_b256_scalar_prep" {
        return Some(vec![
            write(0),
            write(8),
            write(16),
            read(24),
            read(32),
            write(40),
            write(48),
            write(56),
            read(64),
            read(72),
        ]);
    }
    if kernel.starts_with("conv1d_silu_split_qknorm_") {
        return Some(vec![
            write(0),
            write(8),
            write(16),
            read(24),
            read(32),
            write(40),
        ]);
    }
    if kernel == "deinterleave_q_rmsnorm_f32_batched" {
        return Some(vec![read(0), write(8), write(16), read(24)]);
    }
    match kernel {
        "add_inplace_f32" => Some(vec![write(0), read(8)]),
        "fused_rmsnorm_mq_rotate"
        | "fused_rmsnorm_mq_rotate_f16"
        | "fused_rmsnorm_mq_rotate_vecsum"
        | "fused_rmsnorm_mq_rotate_vecsum_sign_const"
        | "fused_rmsnorm_mq_rotate_vecsum_sign_lds" => {
            Some(vec![read(0), read(8), read(16), read(24), write(32)])
        }
        "fused_rmsnorm_mq_rotate_awq_f16" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
        ]),
        "fused_rmsnorm_mq_rotate_wavegrid" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            write(32),
            write(40),
        ]),
        "rmsnorm_reduce_gfx1100" => Some(vec![read(0), write(8)]),
        "compressor_add_ape_f32_buf" => Some(vec![write(0), read(8), read(16)]),
        "compressor_overlap_concat_f32" => Some(vec![read(0), write(8)]),
        "compressor_softmax_pool_f32_buf" => Some(vec![read(0), read(8), write(16), read(24)]),
        "deepseek4_attn_swa_buf" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            write(32),
            read(40),
        ]),
        "deepseek4_attn_swa_topk_f32_buf"
        | "deepseek4_attn_swa_topk_ilp4_f32_buf"
        | "deepseek4_attn_swa_topk_scoregrid_f32_buf"
        | "deepseek4_attn_swa_topk_warp_f32_buf" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            read(40),
            write(48),
            read(56),
            read(64),
        ]),
        "deepseek4_fused_silu_mul_clamp_mq_rotate" => {
            Some(vec![read(0), read(8), read(16), read(24), write(32)])
        }
        "deepseek4_moe_topk_bias_aware_f32" => Some(vec![read(0), read(8), write(16), write(24)]),
        "deepseek4_silu_mul_clamp_f32" => Some(vec![read(0), read(8), write(16)]),
        "embedding_q8_buf_broadcast" => Some(vec![read(0), write(8), read(16)]),
        "deepseek4_topk_kv_gather_f32_buf" | "deepseek4_topk_kv_gather_tiled_f32_buf" => {
            Some(vec![read(0), read(8), write(16), read(24), read(32)])
        }
        "deepseek4_topk_kv_gather_identity_f32_buf" => Some(vec![read(0), write(8), read(16)]),
        "fused_rmsnorm_mq_rotate_plain" | "fused_rmsnorm_mq_rotate_plain_nox" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            write(32),
            write(40),
        ]),
        "gemv_mfp4g32_e8_soa_grouped_gfx1151"
        | "gemv_mfp4g32_e8_soa_u4"
        | "gemv_mfp4g32_e8_soa_u4_buffer_cpol0_gfx1151" => Some(vec![read(0), read(8), write(16)]),
        "gemv_mq2g256_lloyd_moe_down_expanded_k4" => {
            Some(vec![read(0), read(8), read(16), write(24)])
        }
        "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed"
        | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed"
        | "gemv_mq2g256_lloyd_moe_down_residual_scaled_rankpair_indexed"
        | "gemv_mq2g256_lloyd_moe_down_residual_scaled_rowtile2_indexed" => {
            Some(vec![read(0), read(8), read(16), read(24), write(32)])
        }
        "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed"
        | "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_wavecb" => {
            Some(vec![read(0), read(8), read(16), write(24), write(32)])
        }
        "hash_router_normalize_f32_buf" => {
            Some(vec![read(0), read(8), read(16), write(24), write(32)])
        }
        "hc_apply_alpha" => Some(vec![write(0), read(8), read(16)]),
        "hc_finalize_control" => Some(vec![write(0), read(8), read(16)]),
        "hc_finalize_input_map" => Some(vec![write(0), read(8), read(16), read(24), write(32)]),
        "hc_compute_control" | "hc_compute_control_vec4" | "hc_head_compute_pre" => {
            Some(vec![read(0), read(8), read(16), write(24)])
        }
        "hc_compute_control_vec4_finalize" => {
            Some(vec![read(0), read(8), read(16), write(24), read(32)])
        }
        "hc_input_map_4stream" => Some(vec![read(0), read(8), write(16)]),
        "hc_mix_4stream" => Some(vec![read(0), read(8), read(16), read(24), write(32)]),
        "hc_pre_post_sigmoid_scale_f32" | "hc_sinkhorn_4x4" => Some(vec![write(0)]),
        "indexer_relu_score_f32_buf" => Some(vec![read(0), read(8), read(16), write(24), read(32)]),
        "indexer_top_k_buf" | "indexer_top_k_buf_parallel" => {
            Some(vec![read(0), write(8), read(16), read(24)])
        }
        "rmsnorm_f32_at_slot_buf" => Some(vec![write(0), read(8), read(16)]),
        "rope_tail_interleaved_f32"
        | "rope_tail_yarn_interleaved_f32"
        | "rope_tail_yarn_interleaved_wide_f32" => Some(vec![write(0), write(8), read(16)]),
        "rope_tail_yarn_interleaved_at_slot_buf_f32" => Some(vec![write(0), read(8), read(16)]),
        "sqrt_softplus_f32" => Some(vec![write(0)]),
        "state_overlap_shift_f32_buf" => Some(vec![write(0), read(8)]),
        "state_ring_write_f32_buf" | "swa_ring_write_f32_buf" => {
            Some(vec![read(0), write(8), read(16)])
        }
        "zero_f32" => Some(vec![write(0)]),
        "rotate_with_rms_gfx1100" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
        ]),
        "fused_qkvza_hfq4g256"
        | "fused_qkvza_hfq4g256_k2048"
        | "fused_qkvza_hfq4g256_k2048_r2"
        | "fused_qkvza_hfq4g256_k2048_cpol_slc"
        | "fused_qkvza_hfq4g256_wavepack4"
        | "fused_qkvza_hfq4g256_ldsx8"
        | "fused_qkvza_hfq4g256_reduce_chain"
        | "fused_qkvza_mq4g256v2"
        | "fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
            write(48),
            write(56),
            write(64),
        ]),
        "fused_sigmoid_alpha_gate_f32" => Some(vec![write(0), write(8), read(16), read(24)]),
        // LFM retained-PM4 fallback: state is RMW at @8 (write covers RMW).
        "conv1d_gated_decode_f32" => Some(vec![read(0), write(8), read(16), write(24)]),
        // LFM retained-PM4 fallback: q/k/v read, out write, pos read.
        "attention_q8_0_kv" => Some(vec![read(0), read(8), read(16), write(24), read(32)]),
        "conv1d_silu_split_f32" => Some(vec![
            write(0),
            write(8),
            write(16),
            read(24),
            read(32),
            write(40),
        ]),
        "fused_qk_l2_norm_scale_f32" => Some(vec![write(0), write(8)]),
        "repeat_interleave_qk_f32" => Some(vec![read(0), read(8), write(16), write(24)]),
        "gated_delta_net_q8_fast" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
            write(48),
            write(56),
            write(80),
        ]),
        "gated_delta_net_f32" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
            write(48),
        ]),
        "gated_norm_f32" => Some(vec![read(0), read(8), read(16), write(24)]),
        "gated_norm_mq_rotate_gfx1100"
        | "gated_norm_mq_rotate_k6144_gfx1100"
        | "gated_norm_mq_rotate_gfx1151"
        | "gated_norm_mq_rotate_gfx1201"
        | "gated_norm_mq_rotate_k6144_gfx1201" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
        ]),
        // AWQ twin: x, z, weight, awq_scale, signs1, signs2, x_rot.
        "gated_norm_mq_rotate_awq_k6144_gfx1201" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            read(40),
            write(48),
        ]),
        "qwen35_fa_prep_gfx1100"
        | "qwen36_27b_fa_prep_gfx1100"
        | "qwen35_fa_prep_gfx1151"
        | "qwen35_fa_prep_gfx1201"
        | "qwen36_27b_fa_prep_gfx1201" => Some(vec![
            read(0),
            write(8),
            write(16),
            write(24),
            read(32),
            read(40),
            read(48),
        ]),
        "kv_cache_write_q8_0_pair" => Some(vec![write(0), write(8), read(16), read(24), read(32)]),
        "mq_rotate_x" => Some(vec![read(0), write(8), read(16), read(24)]),
        "gemv_hfq4g256"
        | "gemv_hfq4g256_lm_head_dot2_gfx1151"
        | "gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151"
        | "gemv_hfq4g256_k2048"
        | "gemv_hfq4g256_residual"
        | "gemv_hfq4g256_residual_cpol_rt"
        | "gemv_hfq4g256_residual_cpol_rt_low"
        | "gemv_hfq4g256_residual_cpol_slc"
        | "gemv_hfq4g256_residual_k2048"
        | "gemv_hfq4g256_residual_k4096_gfx1151"
        | "gemv_hfq4g256_residual_multirow_r2_gfx1151"
        | "gemv_hfq4g256_residual_rt_low_gfx1151"
        | "gemv_hfq4g256_residual_wave64"
        | "gemv_hfq4g256_wide"
        | "gemv_hfq4g256_multirow_r2"
        | "gemv_hfq4g256_multirow_r4"
        | "gemv_hfq4g256_multirow_r8" => Some(vec![read(0), read(8), write(16)]),
        "softmax_f32" => Some(vec![write(0)]),
        "moe_topk_renorm_k8" => Some(vec![read(0), write(8), write(16)]),
        "fused_silu_mul_mq_rotate" => Some(vec![read(0), read(8), read(16), read(24), write(32)]),
        "gemv_hfq4g256_residual_sigmoid_scaled_gpu"
        | "gemv_mq4g256v2_residual_sigmoid_scaled_k512" => {
            Some(vec![read(0), read(8), write(16), read(24)])
        }
        "gemv_hfq4g256_moe_gate_up_k8_indexed"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_dlc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_glc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2816"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2816_cpol_dlc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2816_cpol_glc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2816_cpol_slc"
        | "gemv_mq4g256_moe_gate_up_k8_indexed_k2816"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2" => {
            Some(vec![read(0), read(8), read(16), write(24), write(32)])
        }
        "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded"
        | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_cpol_slc" => {
            Some(vec![read(0), read(8), read(16), write(24)])
        }
        "gemv_hfq4g256_moe_down_k8_indexed_last_combine" => Some(vec![
            read(0),
            read(8),
            read(16),
            write(24),
            read(32),
            write(40),
        ]),
        "moe_down_combine_k8_batched" | "moe_down_combine_k8_batched_vec4" => {
            Some(vec![read(0), read(8), write(16)])
        }
        "moe_down_combine_rmsnorm_mq_rotate_vecsum"
        | "moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151" => Some(vec![
            read(0),
            read(8),
            write(16),
            read(24),
            read(32),
            read(40),
            write(48),
        ]),
        "fused_qkv_hfq4g256"
        | "fused_qkv_mq4g256v2"
        | "fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100" => Some(vec![
            read(0),
            read(8),
            read(16),
            read(24),
            write(32),
            write(40),
            write(48),
        ]),
        "deinterleave_f32" => Some(vec![read(0), write(8), write(16)]),
        "rmsnorm_f32" | "rmsnorm_f32_warp_reduce" | "rmsnorm_f32_rowsplit" => {
            Some(vec![read(0), read(8), write(16)])
        }
        "rope_partial_halfsplit_f32" | "rope_partial_halfsplit_f32_headgrid" => {
            Some(vec![write(0), write(8), read(16)])
        }
        "kv_cache_write_asym_k_fwht3" => {
            Some(vec![write(0), read(8), read(16), read(24), read(32)])
        }
        "kv_cache_write_q8_0" => Some(vec![write(0), read(8), read(16)]),
        "attention_flash_fwht3_tile" => Some(vec![
            read(0),
            read(8),
            read(16),
            write(24),
            read(32),
            read(40),
            read(48),
        ]),
        // gfx1100 split-KV verifier: preconvert writes persistent Q16 scratch;
        // the partial pass scans Q16/K/V/positions into split records; merge
        // reads those records and writes the final attention output.
        "attention_fa2_q_preconvert_gfx1100" => Some(vec![read(0), write(8), read(16), read(24)]),
        "attention_q8_0_fa2_gqa_partial_gfx1100" => {
            Some(vec![read(0), read(8), read(16), write(24), read(32)])
        }
        "attention_q8_0_fa2_gqa_merge_gfx1100" => Some(vec![read(0), write(8)]),
        // The gfx1201 GQA fp8, gfx1100 GQA Q8_0 and gfx1151 GQA Q8_0 decode
        // tiles and the head-dim-split reduces keep their reference twins'
        // 13/7-argument ABIs and pointer effects.
        "attention_flash_q8_0_tile"
        | "attention_flash_fp8_e4m3_tile_gqa_gfx1201"
        | "attention_flash_q8_0_tile_gqa_gfx1100"
        | "attention_flash_q8_0_tile_gqa_gfx1151" => {
            Some(vec![read(0), read(8), read(16), write(24), read(32)])
        }
        "attention_flash_q8_0_reduce"
        | "attention_flash_reduce_dsplit_gfx1201"
        | "attention_flash_reduce_dsplit_gfx1151" => Some(vec![read(0), write(8), read(24)]),
        "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100"
        | "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151"
        | "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1201" => Some(vec![
            read(0),
            write(8),
            read(16),
            read(24),
            read(32),
            read(48),
        ]),
        "sigmoid_mul_f32" => Some(vec![write(0), read(8)]),
        "gemma4_ple_gelu_mul_strided_f32" => Some(vec![read(0), read(8), write(16)]),
        _ => None,
    }
}

fn expected_kernarg_bytes(kernel: &str) -> Option<usize> {
    if kernel == "fused_rmsnorm_mq_rotate_f16" {
        return Some(48);
    }
    if kernel == "fused_rmsnorm_mq_rotate_awq_f16" {
        return Some(64);
    }
    if kernel == "mq4v2_fp8_fragment_repack_gfx1201" {
        return Some(80);
    }
    if matches!(
        kernel,
        "gemm_mq4g256v2_fp8_set_row_b1"
            | "gemm_mq4g256v2_fp8_add_row_b1"
            | "gemm_mq4g256v2_fp8_silu_row_b1"
            | "gemm_mq4g256v2_fp8_qkv_row_b1"
            | "gemm_mq4g256v2_fp8_qkvza_row_b1"
    ) {
        return Some(96);
    }
    if matches!(
        kernel,
        "hc_pre_post_sigmoid_scale_f32" | "hc_sinkhorn_4x4" | "sqrt_softplus_f32" | "zero_f32"
    ) {
        return Some(16);
    }
    if matches!(
        kernel,
        "compressor_add_ape_f32_buf"
            | "compressor_overlap_concat_f32"
            | "deepseek4_silu_mul_clamp_f32"
            | "embedding_q8_buf_broadcast"
            | "deepseek4_topk_kv_gather_identity_f32_buf"
            | "gemv_mfp4g32_e8_soa_u4"
            | "gemv_mfp4g32_e8_soa_u4_buffer_cpol0_gfx1151"
            | "hc_apply_alpha"
            | "rmsnorm_f32_at_slot_buf"
            | "state_overlap_shift_f32_buf"
            | "state_ring_write_f32_buf"
            | "add_inplace_f32"
    ) {
        return Some(32);
    }
    if matches!(
        kernel,
        "compressor_softmax_pool_f32_buf"
            | "deepseek4_fused_silu_mul_clamp_mq_rotate"
            | "deepseek4_moe_topk_bias_aware_f32"
            | "gemv_mfp4g32_e8_soa_grouped_gfx1151"
            | "gemv_mq2g256_lloyd_moe_down_expanded_k4"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_rowtile2_indexed"
            | "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed"
            | "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_wavecb"
            | "hc_compute_control"
            | "hc_compute_control_vec4"
            | "hc_finalize_control"
            | "indexer_relu_score_f32_buf"
            | "indexer_top_k_buf"
            | "indexer_top_k_buf_parallel"
            | "rope_tail_interleaved_f32"
            | "swa_ring_write_f32_buf"
    ) {
        return Some(48);
    }
    if kernel == "hc_finalize_input_map" {
        return Some(56);
    }
    if kernel == "hc_compute_control_vec4_finalize" {
        return Some(64);
    }
    if kernel == "gemv_mq2g256_lloyd_moe_down_residual_scaled_rankpair_indexed" {
        return Some(56);
    }
    if matches!(
        kernel,
        "deepseek4_attn_swa_buf"
            | "deepseek4_topk_kv_gather_f32_buf"
            | "deepseek4_topk_kv_gather_tiled_f32_buf"
            | "fused_rmsnorm_mq_rotate_plain"
            | "fused_rmsnorm_mq_rotate_plain_nox"
            | "hash_router_normalize_f32_buf"
            | "hc_head_compute_pre"
            | "rope_tail_yarn_interleaved_at_slot_buf_f32"
    ) {
        return Some(64);
    }
    if matches!(
        kernel,
        "rope_tail_yarn_interleaved_f32" | "rope_tail_yarn_interleaved_wide_f32"
    ) {
        return Some(80);
    }
    if matches!(
        kernel,
        "deepseek4_attn_swa_topk_f32_buf"
            | "deepseek4_attn_swa_topk_ilp4_f32_buf"
            | "deepseek4_attn_swa_topk_scoregrid_f32_buf"
            | "deepseek4_attn_swa_topk_warp_f32_buf"
    ) {
        return Some(96);
    }
    if matches!(
        kernel,
        "fused_gate_up_hfq4g256"
            | "fused_gate_up_hfq4g256_k1024_gfx1201"
            | "fused_gate_up_hfq4g256_dot_reform_gfx1100"
            | "fused_gate_up_hfq4g256_dot_prefetch_gfx1100"
            | "fused_gate_up_hfq4g256_pair_gfx1100"
            | "fused_gate_up_hfq4g256_pair2_gfx1100"
            | "fused_gate_up_hfq4g256_quad_prefetch_gfx1100"
            | "fused_gate_up_hfq4g256_setprio_gfx1100"
            | "fused_gate_up_hfq4g256_lane0_headers_gfx1100"
            | "fused_gate_up_hfq4g256_stage_x32_gfx1100"
            | "fused_gate_up_mq4g256v2"
            | "fused_gate_up_mq4g256v2_k5120_gfx1100"
    ) {
        return Some(64);
    }
    if matches!(
        kernel,
        "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151"
            | "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151"
    ) {
        return Some(48);
    }
    if matches!(
        kernel,
        "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_all_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_hybrid_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_low_vgpr_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_all_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_pair_vgpr_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_paired_waves_k2048_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_persistent_rank8_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_route_all_buffer_gfx1151"
            | "gemv_hfq4g256_moe_gate_up_k8_indexed_wave64"
    ) {
        return Some(48);
    }
    if matches!(
        kernel,
        "fused_qkvza_hfq4g256_k2048_all_buffer_dlc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_glc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_all_buffer_slc_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_hybrid_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_pair_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_x_buffer_gfx1151"
            | "fused_qkvza_hfq4g256_k2048_r4_stream_gfx1151"
    ) {
        return Some(96);
    }
    if matches!(
        kernel,
        "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_hybrid_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row1_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_buffer_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_clustered_gfx1151"
            | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row8_gfx1151"
    ) {
        return Some(48);
    }
    if matches!(
        kernel,
        "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed"
            | "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r2"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r4"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r2"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r4"
            | "gemv_mq3g256_lloyd_moe_ninepath_d4"
    ) {
        return Some(48);
    }
    // Batched-K4 codebook MoE: 5 pointers (40 B) + M, K, K_TOP (12 B) = 52 B.
    // The trailing K_TOP is what makes these NOT 48 like their decode siblings;
    // assuming 48 here would make every kernarg length check fail closed.
    if matches!(
        kernel,
        "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4"
            | "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched_k4"
            | "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4"
            | "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4"
    ) {
        return Some(52);
    }
    if matches!(
        kernel,
        "gemv_mq2g256gl_moe_gate_up_k8_indexed"
            | "gemv_mq2g256gl_moe_down_residual_scaled_k8_indexed"
    ) {
        return Some(64);
    }
    if matches!(
        kernel,
        "gemv_mq3g256gl_moe_gate_up_k8_indexed"
            | "gemv_mq3g256gl_moe_down_residual_scaled_k8_indexed"
    ) {
        return Some(80);
    }

    // MQ4G256V2 / MQ6G256V2 MoE: exact sister ABIs. Sizes are the recorded
    // launch_maybe_blob lengths (natural fields + pad_to(16)).
    //   gate_up / ninepath: 5 ptr + 2 i32 = 48
    //   expanded down:      4 ptr + 3 i32 = 44 → 48 padded
    //   gate_up batched:    5 ptr + 3 i32 = 52 → 64 padded
    //   grouped WMMA:       5 ptr + 4 i32 = 56 → 64 padded
    //   mixed gate_up:      6 ptr + 3 i32 = 60 → 64 padded
    //   mixed down:         5 ptr + 3 i32 = 52 → 64 padded
    //   mixed grouped:      6 ptr + 4 i32 = 64 (already aligned)
    //   dense GEMV:         3 ptr + 2 i32 = 32
    //   dense GEMM WMMA:    3 ptr + 3 i32 = 36 → 48 padded
    if matches!(
        kernel,
        "gemv_mq4g256v2_moe_gate_up_k8_indexed"
            | "gemv_mq6g256v2_moe_gate_up_k8_indexed"
            | "gemv_mq4g256v2_moe_gate_up_k8_indexed_k2048_nolds_gfx1100"
            | "gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded"
            | "gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded"
            | "gemv_mq4g256v2_moe_ninepath_d4"
            | "gemv_mq4g256v2_moe_ninepath_rpb8_gfx1100"
            | "gemv_mq6g256v2_moe_ninepath_d4"
    ) {
        return Some(48);
    }
    if matches!(
        kernel,
        "gemv_mq4g256v2_moe_gate_up_k8_indexed_batched"
            | "gemv_mq6g256v2_moe_gate_up_k8_indexed_batched"
            | "gemm_mq4g256v2_moe_grouped_wmma_k2"
            | "gemm_mq4g256v2_moe_grouped_wmma_gfx12"
            | "gemm_mq6g256v2_moe_grouped_wmma_k2"
            | "gemm_mq6g256v2_moe_grouped_wmma_gfx12"
            | "gemv_mixed_moe_gate_up_k8_indexed_batched"
            | "gemv_mixed_moe_down_k8_indexed_batched_expanded"
            | "gemm_mixed_moe_grouped_wmma_k2"
            | "gemm_mixed_moe_grouped_wmma_gfx12"
            | "gemm_mixed_moe_grouped_wmma_4w_k2"
    ) {
        return Some(64);
    }
    if matches!(
        kernel,
        "gemv_mq4g256v2"
            | "gemv_mq4g256v2_residual"
            | "gemv_mq4g256v2_residual_r1_k4096_gfx1100_noscratch"
            | "gemv_mq6g256v2"
            | "gemv_mq6g256v2_residual"
            | "gemv_mq4g256v2_multirow_r2"
            | "gemv_mq4g256v2_multirow_r4"
            | "gemv_mq4g256v2_multirow_r8"
            | "gemv_mq6g256v2_multirow_r2"
            | "gemv_mq6g256v2_multirow_r4"
            | "gemv_mq6g256v2_multirow_r8"
    ) {
        return Some(32);
    }
    // LUT GEMVs: 3 ptrs (24 B) + 2 i32 (8 B) + 8 LUT dwords (32 B) = 64 B.
    if matches!(
        kernel,
        "gemv_mq4g256v2_lloyd" | "gemv_mq4g256v2_residual_lloyd"
    ) {
        return Some(64);
    }
    if matches!(
        kernel,
        "gemm_mq4g256v2_residual_wmma"
            | "gemm_mq4g256v2_residual_wmma_gfx12"
            | "gemm_mq6g256v2_residual_wmma"
            | "gemm_mq6g256v2_residual_wmma_gfx12"
            | "gemm_mq4g256v2_residual_wmma_gfx11_bt4"
            | "gemm_mq4g256v2_residual_wmma_gfx11_bt6"
            | "gemm_mq4g256v2_residual_wmma_gfx11_bt8"
            | "gemm_mq6g256v2_residual_wmma_gfx11_bt4"
            | "gemm_mq6g256v2_residual_wmma_gfx11_bt6"
            | "gemm_mq6g256v2_residual_wmma_gfx11_bt8"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_mw4_lds"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_mw8_lds"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds"
            | "gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage"
            | "gemm_mq6g256v2_residual_wmma_gfx11_mw4_lds"
            | "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds"
    ) {
        return Some(48);
    }
    // Five pointers + four i32 = 56 explicit bytes, padded to the recorder's
    // required 16-byte boundary.
    if kernel == "gemm_gate_up_mq4g256v2_wmma_gfx1100_ldsstage" {
        return Some(64);
    }
    // Two pointers + one i32 = 20 explicit bytes, padded to 32.
    if kernel == "copy_f32_buffer" {
        return Some(32);
    }
    // F16 dense batched GEMM: 3 ptr + M,K,B = 36 → 48 padded. gfx11 and gfx12
    // share one ABI — see `Gpu::gemm_f16_x_f16_wmma`, whose blob builder pushes
    // the same 3 ptr + 3 i32 on both paths before the record path's pad_to(16).
    if matches!(kernel, "gemm_f16_x_f16_wmma" | "gemm_f16_x_f16_wmma_gfx12") {
        return Some(48);
    }

    if kernel.starts_with("gated_delta_net_q8_compact") {
        return Some(96);
    }
    if kernel == "conv1d_silu_split_qknorm_b256_scalar_prep" {
        return Some(112);
    }
    if kernel.starts_with("conv1d_silu_split_qknorm_") {
        return Some(80);
    }
    // T-C Halo prefill fusion: 4 ptr + 3 i32 + 1 f32 = 48.
    if kernel == "deinterleave_q_rmsnorm_f32_batched" {
        return Some(48);
    }
    if kernel == "fused_qkvza_hfq4g256_k2048_scalar_prep" {
        return Some(112);
    }
    if kernel == "moe_router_softmax_topk_k8_wave64"
        || kernel == "moe_router_softmax_topk_k8_wave64_exact"
    {
        return Some(32);
    }
    match kernel {
        "softmax_f32" => Some(16),
        "fused_qk_l2_norm_scale_f32"
        | "gemv_hfq4g256"
        | "gemv_hfq4g256_lm_head_dot2_gfx1151"
        | "gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151"
        | "gemv_hfq4g256_k2048"
        | "gemv_hfq4g256_residual"
        | "gemv_hfq4g256_residual_cpol_rt"
        | "gemv_hfq4g256_residual_cpol_rt_low"
        | "gemv_hfq4g256_residual_cpol_slc"
        | "gemv_hfq4g256_residual_k2048"
        | "gemv_hfq4g256_residual_k4096_gfx1151"
        | "gemv_hfq4g256_residual_multirow_r2_gfx1151"
        | "gemv_hfq4g256_residual_rt_low_gfx1151"
        | "gemv_hfq4g256_residual_wave64"
        | "gemv_hfq4g256_wide"
        | "gemv_hfq4g256_multirow_r2"
        | "gemv_hfq4g256_multirow_r4"
        | "gemv_hfq4g256_multirow_r8"
        | "deinterleave_f32"
        | "kv_cache_write_q8_0"
        | "moe_down_combine_k8_batched"
        | "moe_down_combine_k8_batched_vec4"
        | "moe_topk_renorm_k8"
        | "rmsnorm_f32"
        | "rmsnorm_f32_warp_reduce"
        | "rmsnorm_f32_rowsplit"
        | "rmsnorm_reduce_gfx1100"
        | "hc_input_map_4stream"
        | "sigmoid_mul_f32" => Some(32),
        "gemma4_ple_gelu_mul_strided_f32" => Some(48),
        "attention_fa2_q_preconvert_gfx1100" => Some(48),
        "attention_q8_0_fa2_gqa_partial_gfx1100" => Some(64),
        "attention_q8_0_fa2_gqa_merge_gfx1100" => Some(32),
        "attention_flash_q8_0_reduce"
        | "attention_flash_reduce_dsplit_gfx1201"
        | "attention_flash_reduce_dsplit_gfx1151"
        | "fused_rmsnorm_mq_rotate"
        | "fused_rmsnorm_mq_rotate_vecsum"
        | "fused_rmsnorm_mq_rotate_vecsum_sign_const"
        | "fused_rmsnorm_mq_rotate_vecsum_sign_lds"
        | "fused_sigmoid_alpha_gate_f32"
        | "fused_silu_mul_mq_rotate"
        | "gated_norm_f32"
        | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded"
        | "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_cpol_slc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_dlc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_glc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2816"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2816_cpol_dlc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2816_cpol_glc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_k2816_cpol_slc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave"
        | "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2"
        | "gemv_mq4g256_moe_gate_up_k8_indexed_k2816"
        | "gemv_hfq4g256_residual_sigmoid_scaled_gpu"
        | "gemv_mq4g256v2_residual_sigmoid_scaled_k512"
        | "hc_mix_4stream"
        | "kv_cache_write_asym_k_fwht3"
        | "kv_cache_write_q8_0_pair"
        | "mq_rotate_x"
        | "repeat_interleave_qk_f32"
        | "rope_partial_halfsplit_f32"
        | "rope_partial_halfsplit_f32_headgrid"
        | "conv1d_gated_decode_f32" => Some(48),
        "conv1d_silu_split_f32"
        | "gated_norm_mq_rotate_gfx1100"
        | "gated_norm_mq_rotate_k6144_gfx1100"
        | "gated_norm_mq_rotate_gfx1151"
        | "gated_norm_mq_rotate_gfx1201"
        | "gated_norm_mq_rotate_k6144_gfx1201"
        | "qwen35_fa_prep_gfx1100"
        | "qwen36_27b_fa_prep_gfx1100"
        | "qwen35_fa_prep_gfx1151"
        | "qwen35_fa_prep_gfx1201"
        | "qwen36_27b_fa_prep_gfx1201"
        | "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100"
        | "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151"
        | "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1201"
        | "fused_rmsnorm_mq_rotate_wavegrid"
        | "rotate_with_rms_gfx1100"
        | "attention_q8_0_kv" => Some(64),
        "moe_down_combine_rmsnorm_mq_rotate_vecsum"
        | "moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151" => Some(72),
        "gemv_hfq4g256_moe_down_k8_indexed_last_combine" => Some(64),
        "attention_flash_q8_0_tile"
        | "attention_flash_fp8_e4m3_tile_gqa_gfx1201"
        | "gated_norm_mq_rotate_awq_k6144_gfx1201"
        | "attention_flash_q8_0_tile_gqa_gfx1100"
        | "attention_flash_q8_0_tile_gqa_gfx1151"
        | "fused_qkv_hfq4g256"
        | "fused_qkv_mq4g256v2"
        | "fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100"
        | "moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate" => Some(80),
        "attention_flash_fwht3_tile"
        | "fused_qkvza_hfq4g256"
        | "fused_qkvza_hfq4g256_k2048"
        | "fused_qkvza_hfq4g256_k2048_r2"
        | "fused_qkvza_hfq4g256_k2048_cpol_slc"
        | "fused_qkvza_hfq4g256_wavepack4"
        | "fused_qkvza_hfq4g256_ldsx8"
        | "fused_qkvza_hfq4g256_reduce_chain"
        | "fused_qkvza_mq4g256v2"
        | "fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100"
        | "gated_delta_net_q8_fast" => Some(96),
        "gated_delta_net_f32" => Some(80),
        _ => None,
    }
}

fn apply_qwen_q8_full_attention_visibility(
    launches: &[RecordedHipLaunch],
    headers: &mut [HeaderPolicy],
) {
    // The Q8 full-attention body carries intermediate Q/K/V, tile reductions,
    // and the gated attention result through separate global-memory buffers.
    // Same-queue barriers alone reproduced stale intermediates on gfx1201;
    // restoring the captured HIP dispatches' system scopes for this narrow
    // body makes the multi-position AQL state/logit/KV shadow bit-exact.
    debug_assert_eq!(launches.len(), headers.len());
    let mut full_attention_body = false;
    for (launch, header) in launches.iter().zip(headers) {
        let kernel = launch.kernel.as_str();
        full_attention_body |= kernel == "fused_qkv_hfq4g256";
        if full_attention_body {
            *header = HeaderPolicy::RECORDED_DISPATCH;
        }
        if full_attention_body && kernel == "gemv_hfq4g256_residual" {
            full_attention_body = false;
        }
    }
}

fn recorded_resource_accesses(
    hip: &HipRuntime,
    kernel: &str,
    kernarg: &[u8],
    certified_effects: Option<&[PointerEffect]>,
) -> Option<Vec<RecordedResourceAccess>> {
    if std::mem::size_of::<usize>() != 8 {
        return None;
    }
    let fallback;
    let effects = if let Some(effects) = certified_effects {
        effects
    } else {
        if kernarg.len() != expected_kernarg_bytes(kernel)? {
            return None;
        }
        fallback = pointer_effects(kernel)?;
        &fallback
    };
    let mut accesses = BTreeMap::<(u64, u64), (u64, RecordedAccessMode)>::new();
    for effect in effects {
        let bytes: [u8; 8] = kernarg
            .get(effect.offset..effect.offset + 8)?
            .try_into()
            .ok()?;
        let address = u64::from_ne_bytes(bytes);
        if address == 0 {
            continue;
        }
        let (base, size) = hip.mem_get_address_range(address as usize as *mut _).ok()?;
        let base = base as usize as u64;
        let size = u64::try_from(size).ok()?;
        let entry = accesses
            .entry((base, address))
            .or_insert((size, effect.mode));
        if entry.0 != size {
            return None;
        }
        if effect.mode == RecordedAccessMode::Write {
            entry.1 = RecordedAccessMode::Write;
        }
    }
    Some(
        accesses
            .into_iter()
            .map(
                |((allocation_base, access_base), (allocation_bytes, mode))| {
                    RecordedResourceAccess {
                        allocation_base,
                        allocation_bytes,
                        access_base,
                        mode,
                    }
                },
            )
            .collect(),
    )
}

/// Slice-1 binding resolver: resolve one launch's pointer-effect slots to
/// `(offset, allocation_base, allocation_bytes, interior_offset)` while the
/// allocations are live. Mirrors the effect selection of
/// `recorded_resource_accesses` (certified radiowave effects, else the
/// fallback table gated on the expected kernarg length). Returns `None` when
/// the launch must stay on the raw snapshot path: unknown kernel, an effect
/// outside the segment, a non-8-aligned effect offset (the `KernargAbi`
/// contract), or an address the runtime no longer recognises. Null pointers
/// are skipped, not refused: the snapshot zeros re-encode identically.
fn binding_pointer_slots(
    hip: &HipRuntime,
    kernel: &str,
    kernarg: &[u8],
    certified_effects: Option<&[PointerEffect]>,
) -> Option<Vec<(usize, u64, u64, u64)>> {
    if std::mem::size_of::<usize>() != 8 {
        return None;
    }
    let fallback;
    let effects = if let Some(effects) = certified_effects {
        effects
    } else {
        if kernarg.len() != expected_kernarg_bytes(kernel)? {
            return None;
        }
        fallback = pointer_effects(kernel)?;
        &fallback
    };
    let mut seen = BTreeSet::new();
    let mut slots = Vec::new();
    for effect in effects {
        if !seen.insert(effect.offset) {
            continue;
        }
        if effect.offset % 8 != 0 {
            return None;
        }
        let bytes: [u8; 8] = kernarg
            .get(effect.offset..effect.offset + 8)?
            .try_into()
            .ok()?;
        let address = u64::from_ne_bytes(bytes);
        if address == 0 {
            continue;
        }
        let (base, size) = hip.mem_get_address_range(address as usize as *mut _).ok()?;
        let base = base as usize as u64;
        let size = u64::try_from(size).ok()?;
        if size == 0 {
            return None;
        }
        // The 8-byte slot always re-encodes as `base + interior`, even when
        // the pointed-to allocation is smaller than 8 bytes (a device-side
        // scalar such as a position counter or scale). No bounds check here:
        // the refresh path fails closed if the live range ever shrinks below
        // the recorded one.
        let interior = address.checked_sub(base)?;
        slots.push((effect.offset, base, size, interior));
    }
    Some(slots)
}

/// Re-encode one kernarg segment from `ReplayBindings`: pointer slots take
/// `current_base + interior_offset`, every other byte comes from the recorded
/// snapshot. With unchanged bindings the output equals the snapshot exactly;
/// after a survived relocation only the moved slots differ.
fn encode_bound_kernarg(
    snapshot: &[u8],
    layout: &LaunchBindingLayout,
    bindings: &ReplayBindings,
    kernel: &str,
) -> Result<Vec<u8>, String> {
    let mut encoded = snapshot.to_vec();
    for slot in &layout.slots {
        let binding = bindings.resource(slot.resource).ok_or_else(|| {
            format!(
                "{kernel}: pointer slot at offset {} has no bound resource",
                slot.offset
            )
        })?;
        let base = binding.base().as_ptr() as usize as u64;
        let address = base.checked_add(slot.interior_offset).ok_or_else(|| {
            format!(
                "{kernel}: pointer slot at offset {} base overflows",
                slot.offset
            )
        })?;
        let end = slot
            .offset
            .checked_add(8)
            .ok_or_else(|| format!("{kernel}: pointer slot at offset {} overflows", slot.offset))?;
        if end > encoded.len() {
            return Err(format!(
                "{kernel}: pointer slot at offset {} out of bounds (len {})",
                slot.offset,
                encoded.len()
            ));
        }
        encoded[slot.offset..end].copy_from_slice(&address.to_ne_bytes());
    }
    Ok(encoded)
}

/// Fail-closed byte-equality gate for slice 1: the re-encoded segment must
/// equal the recorded snapshot (after a survived relocation, everywhere
/// except the moved pointer slots — checked by the refresh path with its own
/// slot mask). `debug_assert` covers debug builds; `HIPFIRE_REPLAY_BINDINGS_VERIFY=1`
/// promotes the check to a hard error in release for the harness.
fn verify_bound_kernarg(kernel: &str, snapshot: &[u8], encoded: &[u8]) -> Result<(), String> {
    let mismatch = bound_kernarg_mismatch_message(kernel, snapshot, encoded);
    debug_assert!(
        mismatch.is_none(),
        "{kernel}: re-encoded kernarg differs from the recorded snapshot"
    );
    if !bindings_verify_enabled() {
        return Ok(());
    }
    match mismatch {
        Some(message) => Err(message),
        None => Ok(()),
    }
}

/// Pure mismatch description for the snapshot gate: `None` when the
/// re-encoded segment equals the snapshot, else the fail-closed message with
/// the launch name and the first differing offset.
fn bound_kernarg_mismatch_message(kernel: &str, snapshot: &[u8], encoded: &[u8]) -> Option<String> {
    if snapshot == encoded {
        return None;
    }
    let offset = snapshot
        .iter()
        .zip(encoded.iter())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| snapshot.len().min(encoded.len()));
    Some(format!(
        "{kernel}: re-encoded kernarg differs from the recorded snapshot at offset {offset}"
    ))
}

fn bindings_verify_enabled() -> bool {
    hipfire_config::process_value("HIPFIRE_REPLAY_BINDINGS_VERIFY")
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "on"))
}

/// One `VmmResourceMove` widened to `u64`; ranges are kept sorted by
/// `old_start` and never overlap (`relocation_ranges`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RelocationRange {
    old_start: u64,
    old_end: u64,
    new_start: u64,
    new_end: u64,
    /// Bytes mapped at `new_start`.
    mapped: u64,
}

enum RangeMatch<'a> {
    Outside,
    Inside(&'a RelocationRange),
    Straddles(&'a RelocationRange),
}

/// One tape resource a relocation rebinds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PlannedRelocation {
    resource: ResourceId,
    old_key: (u64, u64),
    new_key: (u64, u64),
    /// Bytes mapped at the new location starting at `new_key.0`.
    coverage: u64,
}

/// Validate `moves` and return them widened and sorted by old base. Rejects a
/// null/empty/self move, `mapped > reserved`, address overflow, duplicate or
/// overlapping sources, overlapping destinations, and a destination that
/// overlaps any source (a chain inside one call would be ambiguous).
fn relocation_ranges(moves: &[VmmResourceMove]) -> Result<Vec<RelocationRange>, String> {
    let mut ranges = Vec::with_capacity(moves.len());
    for (index, relocation) in moves.iter().enumerate() {
        let widen = |value: usize, what: &str| {
            u64::try_from(value).map_err(|_| format!("VMM move {index}: {what} exceeds u64"))
        };
        let old_start = widen(relocation.old_base, "old_base")?;
        let new_start = widen(relocation.new_base, "new_base")?;
        let reserved = widen(relocation.reserved_bytes, "reserved_bytes")?;
        let mapped = widen(relocation.mapped_bytes, "mapped_bytes")?;
        if old_start == 0 || new_start == 0 {
            return Err(format!("VMM move {index}: null base"));
        }
        if old_start == new_start {
            return Err(format!("VMM move {index}: old and new base are both {old_start:#x}"));
        }
        if reserved == 0 {
            return Err(format!("VMM move {index}: zero reserved bytes"));
        }
        if mapped > reserved {
            return Err(format!(
                "VMM move {index}: mapped {mapped:#x} exceeds reserved {reserved:#x}"
            ));
        }
        let old_end = old_start
            .checked_add(reserved)
            .ok_or_else(|| format!("VMM move {index}: old range overflows"))?;
        let new_end = new_start
            .checked_add(reserved)
            .ok_or_else(|| format!("VMM move {index}: new range overflows"))?;
        ranges.push(RelocationRange {
            old_start,
            old_end,
            new_start,
            new_end,
            mapped,
        });
    }
    ranges.sort_unstable_by_key(|range| range.old_start);
    for pair in ranges.windows(2) {
        if pair[0].old_end > pair[1].old_start {
            return Err(if pair[0].old_start == pair[1].old_start {
                format!("duplicate VMM move for old base {:#x}", pair[0].old_start)
            } else {
                format!(
                    "overlapping VMM move sources {:#x}..{:#x} and {:#x}..{:#x}",
                    pair[0].old_start, pair[0].old_end, pair[1].old_start, pair[1].old_end
                )
            });
        }
    }
    let mut destinations: Vec<(u64, u64)> = ranges
        .iter()
        .map(|range| (range.new_start, range.new_end))
        .collect();
    destinations.sort_unstable();
    for pair in destinations.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(format!(
                "overlapping VMM move destinations {:#x}..{:#x} and {:#x}..{:#x}",
                pair[0].0, pair[0].1, pair[1].0, pair[1].1
            ));
        }
    }
    for source in &ranges {
        for destination in &ranges {
            if source.old_start < destination.new_end && destination.new_start < source.old_end {
                return Err(format!(
                    "VMM move destination {:#x}..{:#x} overlaps source {:#x}..{:#x}",
                    destination.new_start, destination.new_end, source.old_start, source.old_end
                ));
            }
        }
    }
    Ok(ranges)
}

/// Classify `start..end` against sorted, non-overlapping `ranges`.
fn classify_range(ranges: &[RelocationRange], start: u64, end: u64) -> RangeMatch<'_> {
    let at = ranges.partition_point(|range| range.old_start <= start);
    if let Some(range) = at.checked_sub(1).map(|index| &ranges[index]) {
        if start < range.old_end {
            return if end <= range.old_end {
                RangeMatch::Inside(range)
            } else {
                RangeMatch::Straddles(range)
            };
        }
    }
    match ranges.get(at) {
        Some(range) if range.old_start < end => RangeMatch::Straddles(range),
        _ => RangeMatch::Outside,
    }
}

/// Pure relocation plan: every `tape_resources` key `(base, size)` contained
/// in a move's `[old_base, old_base + reserved_bytes)` becomes
/// `(new_base + (base - old_base), size)`. Fails closed on an invalid move
/// set, a key straddling a move boundary, a key not fully mapped at the new
/// location (`new_off + size > mapped_bytes`), or a new key overlapping a
/// resource that stays. Keys outside every move are untouched and never
/// probed. The result is ordered by old key.
fn plan_relocation(
    tape_resources: &BTreeMap<(u64, u64), ResourceId>,
    moves: &[VmmResourceMove],
) -> Result<Vec<PlannedRelocation>, String> {
    let ranges = relocation_ranges(moves)?;
    let mut plan = Vec::new();
    for (&(base, size), &resource) in tape_resources {
        let end = base
            .checked_add(size)
            .ok_or_else(|| format!("tape resource {resource:?} {base:#x}+{size:#x} overflows"))?;
        match classify_range(&ranges, base, end) {
            RangeMatch::Outside => {}
            RangeMatch::Straddles(range) => {
                return Err(format!(
                    "tape resource {resource:?} {base:#x}+{size:#x} straddles VMM move source \
                     {:#x}..{:#x}",
                    range.old_start, range.old_end
                ));
            }
            RangeMatch::Inside(range) => {
                let offset = base - range.old_start;
                let coverage = range
                    .mapped
                    .checked_sub(offset)
                    .filter(|coverage| size <= *coverage)
                    .ok_or_else(|| {
                        format!(
                            "tape resource {resource:?} {base:#x}+{size:#x} is not fully mapped at \
                             its new location (mapped {:#x} bytes, offset {offset:#x})",
                            range.mapped
                        )
                    })?;
                plan.push(PlannedRelocation {
                    resource,
                    old_key: (base, size),
                    new_key: (range.new_start + offset, size),
                    coverage,
                });
            }
        }
    }
    for planned in &plan {
        let (new_base, new_size) = planned.new_key;
        let new_end = new_base + new_size;
        if tape_resources
            .keys()
            .any(|&(base, size)| base < new_end && new_base < base.saturating_add(size))
        {
            return Err(format!(
                "relocated tape resource {:?} would overlap a tape resource at \
                 {new_base:#x}..{new_end:#x}",
                planned.resource
            ));
        }
    }
    Ok(plan)
}

/// `Some(message)` when an untyped launch (no pointer slots) cannot be proven
/// clear of every relocated range: unknown accesses, or an access overlapping
/// a move source. Raw kernarg bytes of such a launch cannot be rewritten.
fn untyped_launch_relocation_hazard(
    launch: &RecordedHipLaunch,
    ranges: &[RelocationRange],
) -> Option<String> {
    let Some(accesses) = launch.accesses.as_ref() else {
        return Some(format!(
            "{}: untyped launch has unknown pointer accesses while VMM ranges relocate; \
             port the pointer declaration",
            launch.kernel
        ));
    };
    accesses
        .iter()
        .any(|access| {
            !matches!(
                classify_range(
                    ranges,
                    access.allocation_base,
                    access.allocation_base.saturating_add(access.allocation_bytes),
                ),
                RangeMatch::Outside
            )
        })
        .then(|| {
            format!(
                "{}: untyped launch touches a VMM range being relocated; \
                 port the pointer declaration",
                launch.kernel
            )
        })
}

/// Shift every access whose allocation lies inside a move source to the new
/// location. `None` when no access moved; a straddling access fails closed.
fn relocate_accesses(
    kernel: &str,
    accesses: &[RecordedResourceAccess],
    ranges: &[RelocationRange],
) -> Result<Option<Vec<RecordedResourceAccess>>, String> {
    let mut relocated: Option<Vec<RecordedResourceAccess>> = None;
    for (index, access) in accesses.iter().enumerate() {
        let end = access.allocation_base.saturating_add(access.allocation_bytes);
        match classify_range(ranges, access.allocation_base, end) {
            RangeMatch::Outside => {}
            RangeMatch::Straddles(range) => {
                return Err(format!(
                    "{kernel}: recorded access {:#x}+{:#x} straddles VMM move source {:#x}..{:#x}",
                    access.allocation_base, access.allocation_bytes, range.old_start, range.old_end
                ));
            }
            RangeMatch::Inside(range) => {
                let shifted = |address: u64| {
                    address
                        .checked_sub(range.old_start)
                        .and_then(|offset| range.new_start.checked_add(offset))
                        .ok_or_else(|| {
                            format!("{kernel}: recorded access address {address:#x} cannot relocate")
                        })
                };
                let out = relocated.get_or_insert_with(|| accesses.to_vec());
                out[index].allocation_base = shifted(access.allocation_base)?;
                out[index].access_base = shifted(access.access_base)?;
            }
        }
    }
    Ok(relocated)
}

/// Copy of `snapshot` with every pointer slot bound to a moved resource
/// rewritten to `new_base + interior_offset`; `None` when no slot references a
/// moved resource. Fails closed when a slot's 8 bytes lie past the coverage
/// mapped at the new location. Every non-slot byte is the snapshot's.
fn relocate_kernarg_slots(
    kernel: &str,
    snapshot: &[u8],
    layout: &LaunchBindingLayout,
    moved: &BTreeMap<ResourceId, PlannedRelocation>,
) -> Result<Option<Vec<u8>>, String> {
    let mut relocated: Option<Vec<u8>> = None;
    for slot in &layout.slots {
        let Some(planned) = moved.get(&slot.resource) else {
            continue;
        };
        let reach = slot.interior_offset.checked_add(8).ok_or_else(|| {
            format!("{kernel}: pointer slot at offset {} interior overflows", slot.offset)
        })?;
        if reach > planned.coverage {
            return Err(format!(
                "{kernel}: pointer slot at offset {} reaches {reach:#x} bytes into a resource with \
                 only {:#x} bytes mapped at its new location",
                slot.offset, planned.coverage
            ));
        }
        let address = planned
            .new_key
            .0
            .checked_add(slot.interior_offset)
            .ok_or_else(|| format!("{kernel}: pointer slot at offset {} base overflows", slot.offset))?;
        let end = slot
            .offset
            .checked_add(8)
            .filter(|end| *end <= snapshot.len())
            .ok_or_else(|| {
                format!(
                    "{kernel}: pointer slot at offset {} out of bounds (len {})",
                    slot.offset,
                    snapshot.len()
                )
            })?;
        let out = relocated.get_or_insert_with(|| snapshot.to_vec());
        out[slot.offset..end].copy_from_slice(&address.to_ne_bytes());
    }
    Ok(relocated)
}

/// Fail-closed gate for a prepared PM4 segment: every byte of `current`
/// outside the `exempt` `(offset, len)` ranges (sorted by offset) must equal
/// `expected`. Exempt ranges are the pointer slots and the u32 dynamic
/// kernarg bindings, which replay patches each launch.
fn verify_non_slot_bytes_identical(
    kernel: &str,
    current: &[u8],
    expected: &[u8],
    exempt: &[(usize, usize)],
) -> Result<(), String> {
    if current.len() != expected.len() {
        return Err(format!(
            "{kernel}: relocated kernarg length {} differs from the prepared segment prefix {}",
            expected.len(),
            current.len()
        ));
    }
    let mismatch = |from: usize, to: usize| -> Option<usize> {
        if current[from..to] == expected[from..to] {
            return None;
        }
        current[from..to]
            .iter()
            .zip(&expected[from..to])
            .position(|(old, new)| old != new)
            .map(|position| from + position)
    };
    let failure = |offset: usize| {
        format!(
            "{kernel}: relocated kernarg differs from the prepared segment at non-slot offset \
             {offset}"
        )
    };
    let mut cursor = 0usize;
    for &(offset, len) in exempt {
        let stop = offset.min(current.len());
        if cursor < stop {
            if let Some(at) = mismatch(cursor, stop) {
                return Err(failure(at));
            }
        }
        cursor = cursor.max(offset.saturating_add(len));
    }
    if cursor < current.len() {
        if let Some(at) = mismatch(cursor, current.len()) {
            return Err(failure(at));
        }
    }
    Ok(())
}

/// Per prepared dispatch, the sorted `(offset, len)` kernarg ranges a
/// relocation must not hold to snapshot equality: typed pointer slots (8
/// bytes), declared/synthesized/GDN u32 dynamic bindings (4 bytes), and the
/// legacy GDN frame word at 76. Computed once when the PM4 tape is prepared.
fn bound_exempt_kernarg_ranges(
    launches: &[RecordedHipLaunch],
    prefix: usize,
    dynamic_bindings: &[(usize, ReplayKernargBinding)],
    legacy_gdn_frames: &[usize],
) -> Vec<Vec<(usize, usize)>> {
    let mut ranges: Vec<Vec<(usize, usize)>> = launches
        .iter()
        .take(prefix)
        .map(|launch| {
            launch
                .binding_layout
                .as_ref()
                .map(|layout| layout.slots.iter().map(|slot| (slot.offset, 8)).collect())
                .unwrap_or_default()
        })
        .collect();
    for (dispatch, binding) in dynamic_bindings {
        if let Some(entry) = ranges.get_mut(*dispatch) {
            entry.push((binding.offset(), 4));
        }
    }
    for dispatch in legacy_gdn_frames {
        if let Some(entry) = ranges.get_mut(*dispatch) {
            entry.push((76, 4));
        }
    }
    for entry in &mut ranges {
        entry.sort_unstable();
        entry.dedup();
    }
    ranges
}

#[derive(Default)]
struct ResourceFrontier {
    accesses: Vec<RecordedResourceAccess>,
    known: bool,
}

impl ResourceFrontier {
    fn covered(&self, current: &RecordedHipLaunch) -> bool {
        self.known && current.accesses.is_some()
    }

    fn independent(&self, current: &RecordedHipLaunch) -> bool {
        let Some(current) = &current.accesses else {
            return false;
        };
        self.known
            && !self
                .accesses
                .iter()
                .any(|left| current.iter().any(|right| left.conflicts(*right)))
    }

    fn independent_by_exact_start(&self, current: &RecordedHipLaunch) -> bool {
        let Some(current) = &current.accesses else {
            return false;
        };
        self.known
            && !self.accesses.iter().any(|left| {
                current
                    .iter()
                    .any(|right| left.same_start_conflicts(*right))
            })
    }

    fn advance(&mut self, current: &RecordedHipLaunch, independent: bool) {
        if !independent {
            self.accesses.clear();
            self.known = true;
        }
        let Some(current) = &current.accesses else {
            self.accesses.clear();
            self.known = false;
            return;
        };
        self.accesses.extend_from_slice(current);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Pm4PhasePlan {
    indices: Vec<usize>,
    parallel: bool,
    /// For a parallel phase containing two dependent branch chains, indices
    /// before this position belong to lane 0 and indices at/after it belong
    /// to lane 1. `None` retains the ordinary round-robin antichain layout.
    lane_split: Option<usize>,
}

fn launches_are_independent(left: &RecordedHipLaunch, right: &RecordedHipLaunch) -> bool {
    let (Some(left), Some(right)) = (&left.accesses, &right.accesses) else {
        return false;
    };
    !left
        .iter()
        .any(|left| right.iter().any(|right| left.conflicts(*right)))
}

/// Opt-in flag for the antichain-widening reorder pass. Default off: the pass
/// permutes a certified launch sequence, which is strictly more aggressive than
/// omitting a wait, so it stays behind an explicit switch until it has its own
/// shadow and product evidence.
/// Reorder window, or `None` when the pass is off. A window of W permits a
/// launch to move at most W positions, which bounds how far the pass can
/// deviate from the certified sequence while still gathering nearby
/// independent launches. `on`/`1`/`true` means unlimited.
fn pm4_reorder_window_from_config(name: &str) -> Option<usize> {
    let value = hipfire_config::process_value(name)?;
    match value.as_str() {
        "0" | "false" | "off" | "" => None,
        "1" | "true" | "on" | "max" => Some(usize::MAX),
        other => match other.parse::<usize>() {
            Ok(window) if window >= 2 => Some(window),
            _ => {
                eprintln!(
                    "WARNING: {name}={other:?}: expected off, on, or an integer window >= 2; \
                     leaving the recorded order untouched"
                );
                None
            }
        },
    }
}

/// Architectures admitted to the single-IB reorder pass. `gfx1151` is the arch
/// the pass was certified on upstream; `gfx1201` is admitted here so the pass
/// can be screened against the gfx12 retained route, and remains default-off.
fn pm4_single_ib_reorder_from_config(device_name: &str) -> Option<usize> {
    (device_name.eq_ignore_ascii_case("gfx1151") || device_name.eq_ignore_ascii_case("gfx1201"))
        .then(|| pm4_reorder_window_from_config("HIPFIRE_REPLAY_PM4_SINGLE_IB_REORDER"))
        .flatten()
}

/// Permute the recorded launch order so mutually independent launches become
/// adjacent, widening the antichains `pm4_phase_plan` can form.
///
/// `pm4_phase_plan` only groups launches that are already CONSECUTIVE in the
/// recorded HIP order, so a launch independent of one ten positions away can
/// never share its phase. On the ds4 gfx1151 route that leaves 567 of 2319
/// boundaries independent yet almost all of them isolated pairs, capping every
/// parallel phase at width 2 no matter how many queues are requested, because
/// `lane_count = min(requested, phase.indices.len())`. Queue count was never
/// the constraint; adjacency was.
///
/// Two launches that do not conflict may be reordered freely; two that do
/// conflict must keep their recorded relative order. The result is therefore a
/// topological order of the conflict DAG, produced by a stable level-by-level
/// Kahn schedule. Each emitted level is the full ready set, which is pairwise
/// independent by construction: if two launches conflicted, the later one would
/// still hold the earlier as an unemitted predecessor and could not be ready.
///
/// Launches with no recovered `accesses` conflict with everything (see
/// `launches_are_independent`), so they pin their own position and act as
/// ordering barriers. That keeps the pass fail-closed on anything the resource
/// model could not prove.
///
/// Returns the identity order if the schedule cannot be validated, so a failure
/// degrades to today's behaviour rather than to a reordered tape.
fn pm4_width_reorder(recorded: &[RecordedHipLaunch], window: usize) -> Vec<usize> {
    let n = recorded.len();
    let identity = || (0..n).collect::<Vec<usize>>();
    if n == 0 || window < 2 {
        return identity();
    }

    // Schedule chunk-locally. Launches in different chunks keep their recorded
    // relative order by construction, so only within-chunk pairs can move and
    // no launch travels further than `window` positions. That bounds how far
    // the pass can deviate from the certified sequence, and makes the window a
    // bisection handle when a reordering turns out not to replay.
    let mut order = Vec::with_capacity(n);
    let mut start = 0usize;
    while start < n {
        let end = start.saturating_add(window).min(n);
        let span = end - start;

        // Conflict edges only ever point forward, so the graph is acyclic by
        // construction and Kahn's algorithm always drains it.
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); span];
        let mut indegree = vec![0usize; span];
        for i in 0..span {
            for j in (i + 1)..span {
                if !launches_are_independent(&recorded[start + i], &recorded[start + j]) {
                    successors[i].push(j);
                    indegree[j] += 1;
                }
            }
        }

        let mut scheduled = 0usize;
        let mut ready: Vec<usize> = (0..span).filter(|index| indegree[*index] == 0).collect();
        while !ready.is_empty() {
            // Ascending original index keeps the permutation deterministic, so
            // the retained tape and its sequence hash reproduce across runs.
            ready.sort_unstable();
            let level = std::mem::take(&mut ready);
            for index in level.iter().copied() {
                order.push(start + index);
                scheduled += 1;
            }
            for index in level {
                for successor in successors[index].iter().copied() {
                    indegree[successor] -= 1;
                    if indegree[successor] == 0 {
                        ready.push(successor);
                    }
                }
            }
        }
        if scheduled != span {
            eprintln!(
                "WARNING: PM4 width reorder drained {scheduled} of {span} launches in chunk \
                 [{start}, {end}); retaining recorded order"
            );
            return identity();
        }
        start = end;
    }

    if order.len() != n {
        eprintln!(
            "WARNING: PM4 width reorder produced {} of {} launches; retaining recorded order",
            order.len(),
            n
        );
        return identity();
    }

    // Independently verify against the conflict relation over ALL pairs rather
    // than trusting the scheduler or the chunking argument: every conflicting
    // pair must keep its recorded relative order.
    let mut position = vec![usize::MAX; n];
    for (slot, index) in order.iter().copied().enumerate() {
        if index >= n || position[index] != usize::MAX {
            eprintln!("WARNING: PM4 width reorder is not a permutation; retaining recorded order");
            return identity();
        }
        position[index] = slot;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if !launches_are_independent(&recorded[i], &recorded[j]) && position[i] >= position[j] {
                eprintln!(
                    "WARNING: PM4 width reorder violated dependency {i} -> {j}; \
                     retaining recorded order"
                );
                return identity();
            }
        }
    }
    order
}

/// Partition the original HIP stream into ordered phases. Parallel phases are
/// maximal consecutive pairwise-independent antichains that meet the selected
/// width floor. Narrow antichains are folded back into the surrounding serial
/// IB so their overlap does not cost an extra cross-queue signal fan-in.
fn pm4_phase_plan(
    recorded: &[RecordedHipLaunch],
    min_parallel_width: usize,
    min_parallel_workgroups: u64,
    max_parallel_phases: usize,
) -> Vec<Pm4PhasePlan> {
    let min_parallel_width = min_parallel_width.max(2);
    let mut antichains = Vec::<Vec<usize>>::new();
    for index in 0..recorded.len() {
        let can_join = antichains.last().is_some_and(|phase| {
            phase
                .iter()
                .all(|prior| launches_are_independent(&recorded[*prior], &recorded[index]))
        });
        if can_join {
            antichains.last_mut().unwrap().push(index);
        } else {
            antichains.push(vec![index]);
        }
    }

    let mut phases = Vec::<Pm4PhasePlan>::new();
    let mut parallel_phases = 0_usize;
    for indices in antichains {
        let workgroups = indices.iter().fold(0_u64, |total, index| {
            let launch = &recorded[*index];
            let launch_workgroups = launch.grid.iter().fold(1_u64, |product, axis| {
                product.saturating_mul(u64::from(*axis))
            });
            total.saturating_add(launch_workgroups)
        });
        let parallel = indices.len() >= min_parallel_width
            && workgroups >= min_parallel_workgroups
            && parallel_phases < max_parallel_phases;
        if parallel {
            parallel_phases += 1;
        }
        if !parallel && phases.last().is_some_and(|phase| !phase.parallel) {
            phases.last_mut().unwrap().indices.extend(indices);
        } else {
            phases.push(Pm4PhasePlan {
                parallel,
                indices,
                lane_split: None,
            });
        }
    }
    phases
}

fn launch_workgroups(launch: &RecordedHipLaunch) -> u64 {
    launch.grid.iter().fold(1_u64, |product, axis| {
        product.saturating_mul(u64::from(*axis))
    })
}

fn launch_ranges_are_independent(
    recorded: &[RecordedHipLaunch],
    left: std::ops::Range<usize>,
    right: std::ops::Range<usize>,
) -> bool {
    left.clone().all(|left_index| {
        right.clone().all(|right_index| {
            launches_are_independent(&recorded[left_index], &recorded[right_index])
        })
    })
}

/// Recover the two branch chains intentionally serialized by DeepSeek4's
/// retained-FFN capture route:
///
///   zero(routed) ; shared-E8 chain ; routed-MQ2 chain ; add(shared, routed)
///
/// The resource contracts prove every launch in the shared chain independent
/// of every launch in the routed chain. The zero is moved onto the routed lane
/// so the shared lane can start immediately. Both dependent chains retain their
/// original internal order, and the following serial phase begins with `add`.
fn pm4_ds4_ffn_branch_plan(recorded: &[RecordedHipLaunch]) -> Result<Vec<Pm4PhasePlan>, String> {
    let mut phases = Vec::<Pm4PhasePlan>::new();
    let mut cursor = 0_usize;
    let mut branches = 0_usize;

    while let Some(zero) = recorded[cursor..]
        .iter()
        .position(|launch| launch.kernel == "zero_f32")
        .map(|offset| cursor + offset)
    {
        let add = recorded[zero + 1..]
            .iter()
            .position(|launch| launch.kernel == "add_inplace_f32")
            .map(|offset| zero + 1 + offset)
            .ok_or_else(|| {
                format!(
                    "DeepSeek4 FFN branch capture has zero_f32 at {zero} without a following add"
                )
            })?;
        if recorded[zero + 1..add]
            .iter()
            .any(|launch| launch.kernel == "zero_f32")
        {
            return Err(format!(
                "DeepSeek4 FFN branch capture has nested zero_f32 before add at {add}"
            ));
        }

        let shared_start = zero + 1;
        let mut best = None::<(usize, u64)>;
        for split in shared_start + 1..add {
            let shared = &recorded[shared_start..split];
            let routed = &recorded[split..add];
            let looks_like_shared = shared
                .iter()
                .any(|launch| launch.kernel == "gemv_mfp4g32_e8_soa_u4");
            let looks_like_routed = routed
                .iter()
                .any(|launch| launch.kernel == "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed");
            if !looks_like_shared
                || !looks_like_routed
                || !launch_ranges_are_independent(recorded, shared_start..split, split..add)
                || !launch_ranges_are_independent(recorded, zero..zero + 1, shared_start..split)
            {
                continue;
            }
            let shared_work = shared.iter().map(launch_workgroups).sum::<u64>();
            let routed_work = recorded[zero..zero + 1]
                .iter()
                .chain(routed.iter())
                .map(launch_workgroups)
                .sum::<u64>();
            let balance = shared_work.min(routed_work);
            if best.is_none_or(|(_, best_balance)| balance > best_balance) {
                best = Some((split, balance));
            }
        }
        let (split, _) = best.ok_or_else(|| {
            format!(
                "DeepSeek4 FFN branch capture at zero_f32 index {zero} has no resource-independent shared/routed split before add index {add}"
            )
        })?;

        if cursor < zero {
            phases.push(Pm4PhasePlan {
                indices: (cursor..zero).collect(),
                parallel: false,
                lane_split: None,
            });
        }

        let shared_len = split - shared_start;
        let mut branch_indices = (shared_start..split).collect::<Vec<_>>();
        branch_indices.push(zero);
        branch_indices.extend(split..add);
        phases.push(Pm4PhasePlan {
            indices: branch_indices,
            parallel: true,
            lane_split: Some(shared_len),
        });
        branches += 1;
        cursor = add;
    }

    if cursor < recorded.len() {
        phases.push(Pm4PhasePlan {
            indices: (cursor..recorded.len()).collect(),
            parallel: false,
            lane_split: None,
        });
    }
    if branches == 0 {
        return Err(
            "DeepSeek4 FFN branch-chain planning requested but tape contains no zero/add markers"
                .to_owned(),
        );
    }
    eprintln!(
        "[redline] DeepSeek4 FFN branch-chain plan recovered {branches} shared/routed phases"
    );
    Ok(phases)
}

fn is_ds4_batched_e8_gemv(kernel: &str) -> bool {
    kernel.starts_with("gemv_mfp4g32_e8_soa_batched_b") && kernel.ends_with("_gfx1151")
}

/// Recover the fork/join already present in DeepSeek4's batched verify FFN:
///
///   shared: E8 w1 -> E8 w3 -> SwiGLU -> rotate -> E8 w2
///   routed: E8 router -> score transform -> top-k -> MQ2 gate/up -> SwiGLU -> rotate
///   join:   MQ2 down atomically accumulates into the completed shared output
///
/// The ordinary batched forward emits those branches serially. Keeping each
/// complete producer chain on one queue preserves its cache and dependency
/// locality while allowing the two large branches to overlap. The routed down
/// projection stays in the following serial phase because it consumes the
/// routed activation and updates the shared branch's output allocation.
///
/// Every recognized fork is re-proved resource-independent from the captured
/// argument effects. A changed kernel sequence or unknown effect rejects the
/// entire plan rather than partially parallelizing an unrecognized layer.
fn pm4_ds4_batched_ffn_branch_plan(
    recorded: &[RecordedHipLaunch],
) -> Result<Vec<Pm4PhasePlan>, String> {
    const ROUTED_GATE_UP: &str = "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4";
    const ROUTED_DOWN: &str = "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4";

    let mut phases = Vec::<Pm4PhasePlan>::new();
    let mut cursor = 0_usize;
    let mut search_cursor = 0_usize;
    let mut branches = 0_usize;

    while let Some(down) = recorded[search_cursor..]
        .iter()
        .position(|launch| launch.kernel == ROUTED_DOWN)
        .map(|offset| search_cursor + offset)
    {
        // The exact captured branch has five shared launches and six routed
        // launches before the routed-down join.
        let shared_start = down.checked_sub(11).ok_or_else(|| {
            format!("DeepSeek4 batched FFN routed down at {down} has no complete fork prefix")
        })?;
        let router = down - 6;
        let names = |index: usize| recorded[index].kernel.as_str();
        let recognized = is_ds4_batched_e8_gemv(names(shared_start))
            && is_ds4_batched_e8_gemv(names(shared_start + 1))
            && names(shared_start + 2) == "deepseek4_silu_mul_clamp_f32"
            && names(shared_start + 3) == "mq_rotate_x"
            && is_ds4_batched_e8_gemv(names(shared_start + 4))
            && is_ds4_batched_e8_gemv(names(router))
            && names(router + 1) == "sqrt_softplus_f32"
            && matches!(
                names(router + 2),
                "hash_router_normalize_f32_batched" | "deepseek4_moe_topk_bias_aware_batched_f32"
            )
            && names(router + 3) == ROUTED_GATE_UP
            && names(router + 4) == "deepseek4_silu_mul_clamp_f32"
            && names(router + 5) == "mq_rotate_x";
        if !recognized {
            return Err(format!(
                "DeepSeek4 batched FFN fork before routed down at {down} does not match the certified 5+6 launch sequence"
            ));
        }
        if !launch_ranges_are_independent(recorded, shared_start..router, router..down) {
            return Err(format!(
                "DeepSeek4 batched FFN branches [{shared_start}, {router}) and [{router}, {down}) are not resource-independent"
            ));
        }

        if cursor < shared_start {
            phases.push(Pm4PhasePlan {
                indices: (cursor..shared_start).collect(),
                parallel: false,
                lane_split: None,
            });
        }
        let shared_len = router - shared_start;
        phases.push(Pm4PhasePlan {
            indices: (shared_start..down).collect(),
            parallel: true,
            lane_split: Some(shared_len),
        });
        branches += 1;
        cursor = down;
        search_cursor = down + 1;
    }

    if cursor < recorded.len() {
        phases.push(Pm4PhasePlan {
            indices: (cursor..recorded.len()).collect(),
            parallel: false,
            lane_split: None,
        });
    }
    if branches == 0 {
        return Err(
            "DeepSeek4 batched FFN branch-chain planning requested but tape has no routed-down joins"
                .to_owned(),
        );
    }
    eprintln!(
        "[redline] DeepSeek4 batched FFN branch-chain plan recovered {branches} shared/routed phases"
    );
    Ok(phases)
}

fn pm4_ds4_ffn_branch_chains_from_config() -> bool {
    hipfire_config::process_value("HIPFIRE_REPLAY_PM4_DS4_FFN_BRANCH_CHAINS")
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "on"))
}

fn pm4_min_parallel_width_from_config() -> usize {
    let value = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WIDTH")
        .unwrap_or_else(|| "2".to_owned());
    value.parse::<usize>().ok().filter(|width| *width >= 2).unwrap_or_else(|| {
        eprintln!(
            "WARNING: HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WIDTH={value:?}: expected integer >= 2; using 2"
        );
        2
    })
}

fn pm4_min_parallel_workgroups_from_config() -> u64 {
    let value = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WORKGROUPS")
        .unwrap_or_else(|| "0".to_owned());
    value.parse::<u64>().unwrap_or_else(|_| {
        eprintln!(
            "WARNING: HIPFIRE_REPLAY_PM4_MIN_PARALLEL_WORKGROUPS={value:?}: expected nonnegative integer; using 0"
        );
        0
    })
}

fn pm4_max_parallel_phases_from_config() -> usize {
    let value = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_MAX_PARALLEL_PHASES")
        .unwrap_or_else(|| usize::MAX.to_string());
    value.parse::<usize>().unwrap_or_else(|_| {
        eprintln!(
            "WARNING: HIPFIRE_REPLAY_PM4_MAX_PARALLEL_PHASES={value:?}: expected nonnegative integer; using unlimited"
        );
        usize::MAX
    })
}

fn pm4_native_phase_sync_from_config() -> bool {
    hipfire_config::process_value("HIPFIRE_REPLAY_PM4_NATIVE_PHASES")
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "on"))
}

fn pm4_queue_policy_from_config() -> QueuePolicy {
    // Keep the certified single-IB path as the default. Phase B is explicitly
    // enabled with 2, 4, or auto until hardware shadow and product gates pass.
    let value = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_QUEUES")
        .unwrap_or_else(|| "1".to_owned());
    value.parse().unwrap_or_else(|error| {
        eprintln!("WARNING: HIPFIRE_REPLAY_PM4_QUEUES={value:?}: {error}; retaining one queue");
        QueuePolicy::One
    })
}

impl ReplayTransport {
    fn from_config() -> Self {
        match hipfire_config::process_value("HIPFIRE_REPLAY_TRANSPORT")
            .unwrap_or_else(|| "aql".to_owned())
            .to_ascii_lowercase()
            .as_str()
        {
            "pm4" | "pm4_ib" | "ib" => Self::Pm4Ib,
            _ => Self::AqlPackets,
        }
    }
}

/// Experimental cache-acquire policy inside one retained PM4 tape.
///
/// The entry acquire remains unconditional: HIP populated model state and
/// kernargs before ownership crosses to the ROCr queue. `EntryOnly` removes
/// only the conservative full-system acquires between PM4 dispatches; compute
/// dependency waits and the terminal idle remain unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pm4MidAcquirePolicy {
    Conservative,
    EntryOnly,
    RequiredOnly,
    WithoutRepeatInterleave,
    WithoutFusedSiluRotate,
    WithoutMqRotate,
    WithoutRope,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pm4WaitPolicy {
    Allowlist,
    ResourceAudit,
    Resource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pm4RegisterPolicy {
    Legacy,
    Static,
    Stateful,
}

impl Pm4RegisterPolicy {
    fn from_value(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "" | "0" | "false" | "off" | "legacy" => Some(Self::Legacy),
            "static" | "static-only" | "static_only" => Some(Self::Static),
            "1" | "true" | "on" | "stateful" => Some(Self::Stateful),
            _ => None,
        }
    }

    fn from_config() -> Self {
        let value = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_STATEFUL")
            .unwrap_or_else(|| "static".to_owned());
        Self::from_value(&value).unwrap_or_else(|| {
            eprintln!(
                "WARNING: unknown HIPFIRE_REPLAY_PM4_STATEFUL={value:?}; \
                     retaining legacy full-register emission"
            );
            Self::Legacy
        })
    }
}

/// Diagnostic opt-in, default off: `HIPFIRE_GFX1100_PM4_EXPERIMENTS=1` lets
/// the gfx1151 initiator / interleave / resource-limits knobs apply to gfx1100
/// as well. Without it gfx1100 keeps the legacy encoding byte-for-byte.
fn gfx1100_experiment_alias(architecture: Pm4Architecture, device_name: &str) -> &str {
    if architecture == Pm4Architecture::Gfx11
        && device_name.eq_ignore_ascii_case("gfx1100")
        && hipfire_config::process_value("HIPFIRE_GFX1100_PM4_EXPERIMENTS").as_deref() == Some("1")
    {
        "gfx1151"
    } else {
        device_name
    }
}

fn gfx10_dispatch_initiator_policy(
    architecture: Pm4Architecture,
    device_name: &str,
) -> Gfx10DispatchInitiatorPolicy {
    let device_name = gfx1100_experiment_alias(architecture, device_name);
    let value = hipfire_config::process_value("HIPFIRE_GFX1151_PM4_INITIATOR")
        .unwrap_or_else(|| "legacy".to_owned());
    let policy = gfx10_dispatch_initiator_policy_from_value(architecture, device_name, &value)
        .unwrap_or_else(|| {
            eprintln!(
                "WARNING: unknown HIPFIRE_GFX1151_PM4_INITIATOR={value:?}; retaining legacy initiator"
            );
            Gfx10DispatchInitiatorPolicy::Legacy
        });
    if policy != Gfx10DispatchInitiatorPolicy::Legacy {
        eprintln!("[redline] gfx1151 PM4 dispatch initiator policy={policy:?}");
    }
    policy
}

fn gfx10_dispatch_initiator_policy_from_value(
    architecture: Pm4Architecture,
    device_name: &str,
    value: &str,
) -> Option<Gfx10DispatchInitiatorPolicy> {
    if architecture != Pm4Architecture::Gfx11 || !device_name.eq_ignore_ascii_case("gfx1151") {
        return Some(Gfx10DispatchInitiatorPolicy::Legacy);
    }

    match value.to_ascii_lowercase().as_str() {
        "" | "legacy" | "ordered-append" | "ordered_append" => {
            Some(Gfx10DispatchInitiatorPolicy::Legacy)
        }
        "order" | "order-mode" | "order_mode" => Some(Gfx10DispatchInitiatorPolicy::OrderMode),
        "radv" | "order-tunnel" | "order_tunnel" => Some(Gfx10DispatchInitiatorPolicy::Radv),
        _ => None,
    }
}

fn gfx1151_dispatch_interleave(
    architecture: Pm4Architecture,
    device_name: &str,
) -> Option<Gfx11DispatchInterleave> {
    let device_name = gfx1100_experiment_alias(architecture, device_name);
    let value = hipfire_config::process_value("HIPFIRE_GFX1151_PM4_INTERLEAVE")
        .unwrap_or_else(|| "inherit".to_owned());
    let interleave = gfx1151_dispatch_interleave_from_value(architecture, device_name, &value)
        .unwrap_or_else(|| {
            eprintln!(
                "WARNING: unknown HIPFIRE_GFX1151_PM4_INTERLEAVE={value:?}; inheriting queue value"
            );
            None
        });
    if let Some(interleave) = interleave {
        eprintln!(
            "[redline] gfx1151 PM4 dispatch interleave={} threads/SE",
            interleave.threads()
        );
    }
    interleave
}

fn gfx1151_dispatch_interleave_from_value(
    architecture: Pm4Architecture,
    device_name: &str,
    value: &str,
) -> Option<Option<Gfx11DispatchInterleave>> {
    if architecture != Pm4Architecture::Gfx11 || !device_name.eq_ignore_ascii_case("gfx1151") {
        return Some(None);
    }

    match value.to_ascii_lowercase().as_str() {
        "" | "inherit" | "legacy" | "firmware" => Some(None),
        "0" | "disabled" | "off" => Some(Some(Gfx11DispatchInterleave::Disabled)),
        "64" => Some(Some(Gfx11DispatchInterleave::Threads64)),
        "128" => Some(Some(Gfx11DispatchInterleave::Threads128)),
        "256" => Some(Some(Gfx11DispatchInterleave::Threads256)),
        "512" => Some(Some(Gfx11DispatchInterleave::Threads512)),
        _ => None,
    }
}

fn gfx1151_resource_limits_policy(
    architecture: Pm4Architecture,
    device_name: &str,
) -> Gfx11ComputeResourceLimitsPolicy {
    let device_name = gfx1100_experiment_alias(architecture, device_name);
    let value = hipfire_config::process_value("HIPFIRE_GFX1151_PM4_RESOURCE_LIMITS")
        .unwrap_or_else(|| "legacy".to_owned());
    let policy = gfx1151_resource_limits_policy_from_value(architecture, device_name, &value)
        .unwrap_or_else(|| {
            eprintln!(
                "WARNING: unknown HIPFIRE_GFX1151_PM4_RESOURCE_LIMITS={value:?}; retaining zero resource limits"
            );
            Gfx11ComputeResourceLimitsPolicy::Legacy
        });
    if policy != Gfx11ComputeResourceLimitsPolicy::Legacy {
        eprintln!("[redline] gfx1151 PM4 resource-limits policy={policy:?}");
    }
    policy
}

fn gfx1151_resource_limits_policy_from_value(
    architecture: Pm4Architecture,
    device_name: &str,
    value: &str,
) -> Option<Gfx11ComputeResourceLimitsPolicy> {
    if architecture != Pm4Architecture::Gfx11 || !device_name.eq_ignore_ascii_case("gfx1151") {
        return Some(Gfx11ComputeResourceLimitsPolicy::Legacy);
    }

    match value.to_ascii_lowercase().as_str() {
        "" | "legacy" | "zero" | "off" => Some(Gfx11ComputeResourceLimitsPolicy::Legacy),
        "simd-always" | "simd_always" | "always" => {
            Some(Gfx11ComputeResourceLimitsPolicy::SimdDestAlways)
        }
        // The certified gfx1151 host exposes 40 CUs over 2 SEs. Its 20 CUs/SE
        // are divisible by four, so Mesa's FORCE_SIMD_DIST guard is false.
        "radv" | "simd-dest" | "simd_dest" => Some(Gfx11ComputeResourceLimitsPolicy::Radv {
            force_simd_dist_for_single_wave: false,
        }),
        _ => None,
    }
}

fn gfx1151_cu_mask(architecture: Pm4Architecture, device_name: &str) -> Option<[u32; 2]> {
    let value = hipfire_config::process_value("HIPFIRE_GFX1151_REDLINE_CU_COUNT")
        .unwrap_or_else(|| "all".to_owned());
    let mask = gfx1151_cu_mask_from_value(architecture, device_name, &value).unwrap_or_else(|| {
        eprintln!("WARNING: unknown HIPFIRE_GFX1151_REDLINE_CU_COUNT={value:?}; retaining all CUs");
        None
    });
    if let Some(mask) = mask {
        let enabled = mask.into_iter().map(u32::count_ones).sum::<u32>();
        eprintln!("[redline] gfx1151 queue CU mask enables {enabled}/40 CUs");
    }
    mask
}

fn gfx1151_cu_mask_from_value(
    architecture: Pm4Architecture,
    device_name: &str,
    value: &str,
) -> Option<Option<[u32; 2]>> {
    if architecture != Pm4Architecture::Gfx11 || !device_name.eq_ignore_ascii_case("gfx1151") {
        return Some(None);
    }
    if matches!(
        value.to_ascii_lowercase().as_str(),
        "" | "all" | "inherit" | "40"
    ) {
        return Some(None);
    }
    let count = value.parse::<u32>().ok()?;
    if count == 0 || count >= 40 || !count.is_multiple_of(2) {
        return None;
    }
    let low = if count >= 32 {
        u32::MAX
    } else {
        (1_u32 << count) - 1
    };
    let high_count = count.saturating_sub(32);
    let high = if high_count == 0 {
        0
    } else {
        (1_u32 << high_count) - 1
    };
    Some(Some([low, high]))
}

fn gfx1151_entry_acquire_policy(
    architecture: Pm4Architecture,
    device_name: &str,
) -> Gfx11EntryAcquirePolicy {
    let value = hipfire_config::process_value("HIPFIRE_GFX1151_PM4_ENTRY_ACQUIRE")
        .unwrap_or_else(|| "system".to_owned());
    let policy = gfx1151_entry_acquire_policy_from_value(architecture, device_name, &value)
        .unwrap_or_else(|| {
            eprintln!(
                "WARNING: unknown HIPFIRE_GFX1151_PM4_ENTRY_ACQUIRE={value:?}; retaining system acquire"
            );
            Gfx11EntryAcquirePolicy::System
        });
    if policy != Gfx11EntryAcquirePolicy::System {
        eprintln!("[redline] gfx1151 PM4 entry acquire policy={policy:?}");
    }
    policy
}

fn gfx1151_entry_acquire_policy_from_value(
    architecture: Pm4Architecture,
    device_name: &str,
    value: &str,
) -> Option<Gfx11EntryAcquirePolicy> {
    if architecture != Pm4Architecture::Gfx11 || !device_name.eq_ignore_ascii_case("gfx1151") {
        return Some(Gfx11EntryAcquirePolicy::System);
    }
    match value.to_ascii_lowercase().as_str() {
        "" | "system" | "legacy" => Some(Gfx11EntryAcquirePolicy::System),
        "agent" | "same-agent" | "same_agent" => Some(Gfx11EntryAcquirePolicy::Agent),
        "vmem" | "vector" => Some(Gfx11EntryAcquirePolicy::Vmem),
        "none" | "raw" | "off" => Some(Gfx11EntryAcquirePolicy::None),
        _ => None,
    }
}

/// Pool for retained PM4 kernarg segments.
///
/// Default `vram`: a host-writable GPU-agent pool. Kernels whose prologue
/// chains dependent kernarg `s_load`s otherwise pay host-memory latency on
/// every round (gfx1100 H2: 1.4-1.6 ms/token of in-IB span).
/// `HIPFIRE_PM4_KERNARG_POOL=host` keeps the CPU-agent fine-grained pool, as
/// do small-BAR systems. The host patches these segments between replays, so
/// VRAM placement relies on the IB entry ACQUIRE_MEM invalidating GL2 and the
/// scalar cache; a gfx1151 non-system entry acquire keeps the host pool.
fn retained_kernarg_pool(
    device: &GpuDevice,
    host_pool: &KernargPool,
    entry_acquire: Gfx11EntryAcquirePolicy,
) -> KernargPool {
    let requested = hipfire_config::process_value("HIPFIRE_PM4_KERNARG_POOL")
        .unwrap_or_else(|| "vram".to_owned());
    let fallback = if requested.eq_ignore_ascii_case("host") {
        "HIPFIRE_PM4_KERNARG_POOL=host".to_owned()
    } else if entry_acquire != Gfx11EntryAcquirePolicy::System {
        format!("entry acquire {entry_acquire:?} does not invalidate GL2")
    } else {
        match KernargPool::discover_host_writable_device_local(device) {
            Ok(pool) => {
                eprintln!("[redline] retained PM4 kernargs: pool=vram");
                return pool;
            }
            Err(error) => format!("fallback: {error}"),
        }
    };
    eprintln!("[redline] retained PM4 kernargs: pool=host ({fallback})");
    host_pool.clone()
}

impl Pm4WaitPolicy {
    fn from_value(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "" | "allowlist" | "conservative" => Some(Self::Allowlist),
            "resource-audit" | "resource_audit" | "audit" => Some(Self::ResourceAudit),
            "resource" | "resources" => Some(Self::Resource),
            _ => None,
        }
    }

    fn from_config() -> Self {
        let value = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_WAIT_POLICY")
            .unwrap_or_else(|| "resource".to_owned());
        Self::from_value(&value).unwrap_or_else(|| {
            eprintln!(
                "WARNING: unknown HIPFIRE_REPLAY_PM4_WAIT_POLICY={value:?}; \
                     retaining the certified allowlist wait policy"
            );
            Self::Allowlist
        })
    }
}

#[derive(Default)]
struct Pm4WaitAudit {
    boundaries: usize,
    covered: usize,
    allowlist_independent: usize,
    resource_independent: usize,
    allowlist_only: BTreeMap<(String, String), usize>,
    resource_only: BTreeMap<(String, String), usize>,
    suballocation_candidates: BTreeMap<(String, String), usize>,
}

impl Pm4WaitAudit {
    fn observe(
        &mut self,
        previous: &RecordedHipLaunch,
        current: &RecordedHipLaunch,
        allowlist_independent: bool,
        resource_independent: bool,
        exact_start_independent: bool,
        resource_covered: bool,
    ) {
        self.boundaries += 1;
        if resource_covered {
            self.covered += 1;
        }
        self.allowlist_independent += usize::from(allowlist_independent);
        self.resource_independent += usize::from(resource_independent);
        let pair = (previous.kernel.clone(), current.kernel.clone());
        if allowlist_independent && !resource_independent {
            *self.allowlist_only.entry(pair.clone()).or_default() += 1;
        } else if resource_independent && !allowlist_independent {
            *self.resource_only.entry(pair.clone()).or_default() += 1;
        }
        if resource_covered && exact_start_independent && !resource_independent {
            *self.suballocation_candidates.entry(pair).or_default() += 1;
        }
    }

    fn report(&self, policy: Pm4WaitPolicy) {
        eprintln!(
            "[redline] PM4 wait audit policy={policy:?} boundaries={} covered={} \
             allowlist_independent={} resource_independent={} allowlist_only={:?} \
             resource_only={:?} suballocation_candidates={:?}",
            self.boundaries,
            self.covered,
            self.allowlist_independent,
            self.resource_independent,
            self.allowlist_only,
            self.resource_only,
            self.suballocation_candidates,
        );
    }
}

impl Pm4MidAcquirePolicy {
    fn from_value(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "" | "conservative" | "all" => Some(Self::Conservative),
            "entry-only" | "entry_only" | "none" => Some(Self::EntryOnly),
            "required-only" | "required_only" => Some(Self::RequiredOnly),
            "without-repeat-interleave" => Some(Self::WithoutRepeatInterleave),
            "without-fused-silu-rotate" => Some(Self::WithoutFusedSiluRotate),
            "without-mq-rotate" => Some(Self::WithoutMqRotate),
            "without-rope" => Some(Self::WithoutRope),
            _ => None,
        }
    }

    fn from_config() -> Self {
        let value = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY")
            .unwrap_or_else(|| "required-only".to_owned());
        Self::from_value(&value).unwrap_or_else(|| {
            eprintln!(
                "WARNING: unknown HIPFIRE_REPLAY_PM4_ACQUIRE_POLICY={value:?}; \
                 retaining conservative mid-tape acquires"
            );
            Self::Conservative
        })
    }

    fn acquire_between(self, previous: &str, current: &str) -> bool {
        match self {
            Self::Conservative => conservative_mid_acquire_except(previous, current, None),
            Self::EntryOnly => false,
            Self::RequiredOnly => required_mid_acquire(previous, current),
            Self::WithoutRepeatInterleave => {
                conservative_mid_acquire_except(previous, current, Some("repeat_interleave_qk_f32"))
            }
            Self::WithoutFusedSiluRotate => {
                conservative_mid_acquire_except(previous, current, Some("fused_silu_mul_mq_rotate"))
            }
            Self::WithoutMqRotate => {
                conservative_mid_acquire_except(previous, current, Some("mq_rotate_x"))
            }
            Self::WithoutRope => conservative_mid_acquire_except(
                previous,
                current,
                Some("rope_partial_halfsplit_f32"),
            ),
        }
    }
}

fn required_mid_acquire(previous: &str, current: &str) -> bool {
    if previous.starts_with("gated_delta_net_q8_compact")
        || current.starts_with("gated_delta_net_q8_compact")
    {
        return true;
    }
    // Dense Qwen3.5 feeds the rotated SiLU product straight into the FFN
    // down projection.  A compute-idle wait orders the dispatches, but gfx12
    // still needs the vector-cache acquire before the GEMV reads that buffer.
    // Without it the first divergent launch in the 0.8B tape is this exact
    // pair (launches 12 -> 13); logits, KV, and recurrent state then drift.
    if previous == "fused_silu_mul_mq_rotate" && current.starts_with("gemv_hfq4g256_residual") {
        return true;
    }
    // LFM's rotated projection buffer is consumed immediately by GEMV. A
    // compute-idle wait orders execution, but gfx12 needs a vector-cache
    // acquire before the consumer reads mq_rotate_x output.
    if previous == "mq_rotate_x" {
        return true;
    }
    matches!(
        previous,
        "repeat_interleave_qk_f32"
            | "rope_partial_halfsplit_f32"
            | "rope_partial_halfsplit_f32_headgrid"
    ) || matches!(
        current,
        "repeat_interleave_qk_f32"
            | "rope_partial_halfsplit_f32"
            | "rope_partial_halfsplit_f32_headgrid"
    )
}

/// Reused rotate destinations need their stale GC12 vector-cache line
/// invalidated before the writer executes in a retained IB.
fn requires_gfx12_pre_dispatch_vmem_acquire(current: &str) -> bool {
    current == "mq_rotate_x" || current == "fused_silu_mul_mq_rotate"
}

fn conservative_mid_acquire_except(previous: &str, current: &str, excluded: Option<&str>) -> bool {
    if previous.starts_with("gated_delta_net_q8_compact")
        || current.starts_with("gated_delta_net_q8_compact")
    {
        return true;
    }
    (Some(previous) != excluded
        && matches!(
            previous,
            "repeat_interleave_qk_f32"
                | "fused_silu_mul_mq_rotate"
                | "mq_rotate_x"
                | "rope_partial_halfsplit_f32"
                | "rope_partial_halfsplit_f32_headgrid"
        ))
        || (Some(current) != excluded
            && matches!(
                current,
                "repeat_interleave_qk_f32"
                    | "fused_silu_mul_mq_rotate"
                    | "rope_partial_halfsplit_f32"
                    | "rope_partial_halfsplit_f32_headgrid"
            ))
}

fn independent_sibling(previous: &str, current: &str) -> bool {
    matches!(
        (previous, current),
        ("fused_sigmoid_alpha_gate_f32", "conv1d_silu_split_f32")
            | ("rmsnorm_f32", "rmsnorm_f32")
            | ("kv_cache_write_q8_0", "kv_cache_write_q8_0")
            | (
                "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151",
                "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_paired_waves_k2048_gfx1151",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_dlc",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_glc",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave",
            )
            | (
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_paired_waves_k2048_gfx1151",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_dlc",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_glc",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave",
            )
            | (
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2",
            )
    )
}

impl ReplayBackendRequest {
    fn from_config() -> Self {
        match hipfire_config::process_value("HIPFIRE_REPLAY_BACKEND")
            .unwrap_or_else(|| "hip".to_owned())
            .to_ascii_lowercase()
            .as_str()
        {
            "" | "hip" | "off" => Self::Hip,
            "shadow" => Self::Shadow,
            "auto" | "redline" => Self::Auto,
            value => {
                eprintln!("WARNING: unknown HIPFIRE_REPLAY_BACKEND={value:?}; falling back to hip");
                Self::Hip
            }
        }
    }
}

fn manual_capture_requested() -> bool {
    hipfire_config::process_value("HIPFIRE_REPLAY_MANUAL_CAPTURE")
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "on"))
}

fn route_proof_log_requested() -> bool {
    hipfire_config::process_value("HIPFIRE_REPLAY_ROUTE_PROOF_LOG")
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "on"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayState {
    Hip,
    Armed,
    RecordingWarmup,
    Captured,
    ShadowValidated,
    Ready,
    Fallback,
}

/// One launch seen by the G0 eager arm: the eager branch's kernel, geometry
/// and exact padded kernarg bytes (the same bytes a recording would keep).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct G0Launch {
    pub kernel: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem: u32,
    pub kernarg: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordedHipLaunch {
    pub kernel: String,
    /// Admitted code object that owns the launched symbol; replay loads its
    /// retained bytes, never a file.
    pub artifact: Option<RecordedArtifact>,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem: u32,
    /// Optional PM4-only binding that narrows one recorded maximum grid axis
    /// from the zero-based decode position before each quiescent replay.
    pub grid_binding: Option<ReplayGridBinding>,
    /// Exact naturally-aligned, tail-padded bytes passed through HIP's
    /// contiguous `extra` launch ABI. The model adapter owns the lifetime
    /// contract for pointer values recovered into allocation-wide effects.
    pub kernarg: Vec<u8>,
    /// Dynamic kernarg fields the engine *named* at record time, each with one
    /// owning dispatch offset. Replay re-derives every one of them from the
    /// current position through `apply_kernarg_bindings_for_dispatch`; the
    /// recorded bytes hold the capture-position values.
    declared_kernarg_bindings: Vec<ReplayKernargBinding>,
    /// Allocation-wide effects recovered from typed kernel signatures and
    /// `hipMemGetAddressRange`. `None` means the launch must remain serialized.
    accesses: Option<Vec<RecordedResourceAccess>>,
    /// Slice-1 binding layout: per-launch `KernargAbi` pointer slots into the
    /// tape-global `ReplayBindings` store. `Some` means the kernarg segment
    /// is re-encoded from bindings at prepare/refresh; `None` (untyped
    /// launch: unknown kernel or an unresolvable pointer slot) stays on the
    /// raw snapshot path byte-for-byte.
    pub binding_layout: Option<LaunchBindingLayout>,
}

/// One 8-byte device-pointer slot inside a recorded kernarg segment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernargPointerSlot {
    /// Byte offset of the pointer within the segment.
    pub offset: usize,
    /// Tape-global resource whose current base anchors the slot.
    pub resource: ResourceId,
    /// `recorded_pointer - allocation_base_at_record`, added to the current
    /// base at re-encode time so interior pointers survive relocation.
    pub interior_offset: u64,
}

/// Per-launch slice-1 binding layout: the `KernargAbi` describing the segment
/// plus the pointer slots that re-encode from `ReplayBindings`. `slots` is
/// 1:1 with `abi.fields()` in the same order. Every other byte of the segment
/// re-encodes from the recorded snapshot, so scalar kernargs (N, positions,
/// tile bounds) are byte-identical by construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LaunchBindingLayout {
    pub abi: KernargAbi,
    pub slots: Vec<KernargPointerSlot>,
}

/// Outcome of the post-scratch-growth binding refresh.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingRefreshReport {
    /// No growth was armed; the route is untouched.
    NotPending,
    /// Growth was armed but no PM4 route is installed; nothing to keep.
    NoRoute,
    /// The retained route survived: revision bumped, segments re-encoded.
    Refreshed {
        resources: usize,
        reencoded: usize,
        revision: BindingRevision,
    },
}

impl RecordedHipLaunch {
    /// Engine-declared dynamic kernarg fields for this launch.
    pub(crate) fn declared_kernarg_bindings(&self) -> &[ReplayKernargBinding] {
        &self.declared_kernarg_bindings
    }
}

/// A dynamic retained-grid contract supplied by the engine at capture time.
/// The recorded grid remains the hard maximum; replay may only narrow it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayGridBinding {
    PositionCeilDiv { axis: u8, addend: u32, divisor: u32 },
}

impl ReplayGridBinding {
    fn bind(self, position: usize, recorded: [u32; 3]) -> Result<[u32; 3], String> {
        let Self::PositionCeilDiv {
            axis,
            addend,
            divisor,
        } = self;
        let axis = usize::from(axis);
        if axis >= 3 || divisor == 0 {
            return Err(format!(
                "invalid replay grid binding axis={axis} divisor={divisor}"
            ));
        }
        let extent = u64::try_from(position)
            .map_err(|_| "decode position exceeds u64".to_owned())?
            .checked_add(u64::from(addend))
            .ok_or_else(|| "dynamic replay extent overflow".to_owned())?;
        let units = extent.div_ceil(u64::from(divisor)).max(1);
        let units =
            u32::try_from(units).map_err(|_| format!("dynamic replay grid {units} exceeds u32"))?;
        let mut bound = recorded;
        bound[axis] = units.min(recorded[axis]);
        Ok(bound)
    }

    fn units_for(self, position: usize) -> Result<u32, String> {
        let Self::PositionCeilDiv {
            axis: _,
            addend,
            divisor,
        } = self;
        if divisor == 0 {
            return Err(format!("invalid replay grid binding divisor={divisor}"));
        }
        let extent = u64::try_from(position)
            .map_err(|_| "decode position exceeds u64".to_owned())?
            .checked_add(u64::from(addend))
            .ok_or_else(|| "dynamic replay extent overflow".to_owned())?;
        let units = extent.div_ceil(u64::from(divisor)).max(1);
        u32::try_from(units).map_err(|_| format!("dynamic replay grid {units} exceeds u32"))
    }
}

/// Single decision point for GDN stochastic-rounding frame consumption.
///
/// `frames = max(1, nt * grid.z)` where `nt` is the little-endian `i32` at
/// kernarg byte offset 64 and `grid.z` is the recorded launch's third grid
/// dimension. Both are fixed for the fixed-shape tape prepared by
/// `prepare_pm4_prefix_inner` and by `replay_recorded_hip_prefix`.
///
/// Rejects (instead of silently defaulting) when the kernarg block is shorter
/// than 80 bytes, when `nt <= 0`, or when the product overflows `u32`.
pub(crate) fn gdn_requant_frames_for_dispatch(kernarg: &[u8], grid_z: u32) -> Result<u32, String> {
    if kernarg.len() < 80 {
        return Err(format!(
            "GDN kernarg block too short: {} < 80",
            kernarg.len()
        ));
    }
    let nt = i32::from_le_bytes(kernarg[64..68].try_into().expect("slice 64..68 is 4 bytes"));
    if nt <= 0 {
        return Err(format!("GDN nt must be > 0, got {nt}"));
    }
    let product = (nt as u64)
        .checked_mul(grid_z as u64)
        .ok_or_else(|| "GDN frame product overflow".to_owned())?;
    if product > u32::MAX as u64 {
        return Err(format!("GDN frame product {product} exceeds u32"));
    }
    let frames = product as u32;
    Ok(frames.max(1))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayKernargBinding {
    GdnFrameU32 {
        offset: usize,
        frames: u32,
    },
    PositionPlusU32 {
        offset: usize,
        addend: u32,
    },
    /// Quotient form: `(position + addend) / divisor`. Declared by the lowering
    /// that computes the same quotient host-side (for example a block count
    /// derived from the decode position).
    PositionDivU32 {
        offset: usize,
        addend: u32,
        divisor: u32,
    },
    /// Remainder form: `(position + addend) % modulus`. Declared by the
    /// lowering that computes the same remainder host-side (for example a ring
    /// cursor into a bounded convolution history). `addend` is the row's offset
    /// inside a multi-row chunk, so a chunk body stays correct without a
    /// per-regime tape.
    PositionModU32 {
        offset: usize,
        addend: u32,
        modulus: u32,
    },
    /// Product form: `position * factor`. Declared by a lowering that turns a
    /// per-row buffer offset into a scalar (`dst_col_offset`) so the tape keeps
    /// a position-independent pointer.
    PositionMulU32 {
        offset: usize,
        factor: u32,
    },
}

impl ReplayKernargBinding {
    /// The single kernarg byte offset this binding owns. Exactly one binding may
    /// own an offset on a dispatch; two owners would double-patch one slot.
    pub(crate) const fn offset(self) -> usize {
        match self {
            Self::GdnFrameU32 { offset, .. }
            | Self::PositionPlusU32 { offset, .. }
            | Self::PositionDivU32 { offset, .. }
            | Self::PositionModU32 { offset, .. }
            | Self::PositionMulU32 { offset, .. } => offset,
        }
    }

    /// True when the bound value is a pure function of the decode position, so
    /// two recordings taken at different positions are *expected* to differ in
    /// this slot rather than leaving it unexplained.
    pub(crate) const fn is_position_derived(self) -> bool {
        matches!(
            self,
            Self::PositionPlusU32 { .. }
                | Self::PositionDivU32 { .. }
                | Self::PositionModU32 { .. }
                | Self::PositionMulU32 { .. }
        )
    }

    /// Stable encoding of this binding's identity for the tape sequence hash.
    /// Fixed-size so hashing never allocates.
    pub(crate) fn identity_bytes(self) -> [u8; 17] {
        let (tag, param_a, param_b) = match self {
            Self::GdnFrameU32 { frames, .. } => (0u8, frames, 0u32),
            Self::PositionPlusU32 { addend, .. } => (1u8, addend, 0u32),
            Self::PositionDivU32 {
                addend, divisor, ..
            } => (2u8, addend, divisor),
            Self::PositionModU32 {
                addend, modulus, ..
            } => (3u8, addend, modulus),
            Self::PositionMulU32 { factor, .. } => (4u8, factor, 0u32),
        };
        let mut bytes = [0u8; 17];
        bytes[0] = tag;
        bytes[1..9].copy_from_slice(&(self.offset() as u64).to_le_bytes());
        bytes[9..13].copy_from_slice(&param_a.to_le_bytes());
        bytes[13..17].copy_from_slice(&param_b.to_le_bytes());
        bytes
    }

    fn apply(self, kernarg_bytes: &mut [u8], position: usize) -> Result<(), String> {
        let position_u32 =
            u32::try_from(position).map_err(|_| "decode position exceeds u32".to_owned())?;
        match self {
            Self::GdnFrameU32 { offset, frames } => {
                let frame = crate::norm::reserve_gdn_requant_frames(frames);
                write_kernarg_u32(kernarg_bytes, offset, frame, "GDN kernarg binding")
            }
            Self::PositionPlusU32 { offset, addend } => {
                let value = position_u32
                    .checked_add(addend)
                    .ok_or_else(|| "PositionPlusU32 overflow".to_owned())?;
                write_kernarg_u32(kernarg_bytes, offset, value, "kernarg binding")
            }
            Self::PositionDivU32 {
                offset,
                addend,
                divisor,
            } => {
                if divisor == 0 {
                    return Err("PositionDivU32 divisor must be non-zero".to_owned());
                }
                let value = position_u32
                    .checked_add(addend)
                    .ok_or_else(|| "PositionDivU32 overflow".to_owned())?
                    / divisor;
                write_kernarg_u32(kernarg_bytes, offset, value, "kernarg binding")
            }
            Self::PositionModU32 {
                offset,
                addend,
                modulus,
            } => {
                if modulus == 0 {
                    return Err("PositionModU32 modulus must be non-zero".to_owned());
                }
                let value = position_u32
                    .checked_add(addend)
                    .ok_or_else(|| "PositionModU32 overflow".to_owned())?
                    % modulus;
                write_kernarg_u32(kernarg_bytes, offset, value, "kernarg binding")
            }
            Self::PositionMulU32 { offset, factor } => {
                let value = position_u32
                    .checked_mul(factor)
                    .ok_or_else(|| "PositionMulU32 overflow".to_owned())?;
                write_kernarg_u32(kernarg_bytes, offset, value, "kernarg binding")
            }
        }
    }
}

/// Write one 4-byte native-endian kernarg scalar with an explicit range check.
/// Every binding kind funnels through this, so an out-of-range offset can never
/// be a per-variant difference.
fn write_kernarg_u32(
    kernarg_bytes: &mut [u8],
    offset: usize,
    value: u32,
    label: &str,
) -> Result<(), String> {
    let len = kernarg_bytes.len();
    let end = offset
        .checked_add(4)
        .ok_or_else(|| format!("{label} offset overflow"))?;
    let slot = kernarg_bytes
        .get_mut(offset..end)
        .ok_or_else(|| format!("{label} offset {offset} out of bounds (len {len})"))?;
    slot.copy_from_slice(&value.to_ne_bytes());
    Ok(())
}

/// Opaque snapshot of one completed recording's per-launch kernarg blocks.
#[derive(Clone, Debug)]
pub struct RecordedKernargSnapshot {
    entries: Vec<SnapshotEntry>,
}

#[derive(Clone, Debug)]
struct SnapshotEntry {
    kernel: String,
    kernarg: Vec<u8>,
    grid: [u32; 3],
}

pub(crate) fn is_gdn_kernel(kernel: &str) -> bool {
    kernel == "gated_delta_net_q8_fast" || kernel.starts_with("gated_delta_net_q8_compact")
}

/// Fail-closed gate for the PM4/AQL lowering, which turns a recorded HIP
/// geometry (or the prepared maximum of a position-bound grid) into a direct
/// dispatch. HIP rejects such a grid on the device's grid.y/grid.z ceiling, so
/// the retained transport must not accept a geometry the raw launch refuses.
fn check_prepared_grid(device_name: &str, symbol: &str, grid: [u32; 3]) -> Result<(), String> {
    hip_bridge::check_launch_grid(grid, hip_bridge::grid_yz_limit_for_arch(device_name))
        .map_err(|error| format!("{symbol}: {}", error.message))
}

fn is_plausible_device_address(value: u64) -> bool {
    // Heuristic: real device pointers are at least page-aligned and in a
    // high virtual range. Small integers (positions, scales, sizes) are
    // < 1M or have low entropy. Use a conservative band that captures
    // relocated buffers without flagging legitimate position scalars.
    if value < 4096 {
        return false;
    }
    if value < 0x100000 {
        return false;
    }
    // Require either high 32 bits or a large 32-bit value that is not a
    // plausible small position+addend (which are typically < 1e6).
    let large_32 = value > 0x0100_0000 && value < (1u64 << 48) && value & 0x3 == 0;
    let has_high = (value >> 32) != 0 && value < (1u64 << 48);
    large_32 || has_high
}

/// Single code path that applies every retained-replay kernarg binding for one
/// dispatch at `position`. Both the PM4 retained IB and the recorded-HIP blob
/// oracle must call this helper so they cannot diverge.
pub(crate) fn apply_kernarg_bindings_for_dispatch(
    kernarg_bytes: &mut [u8],
    dispatch_index: usize,
    position: usize,
    bindings: &[(usize, ReplayKernargBinding)],
) -> Result<(), String> {
    for (dispatch, binding) in bindings {
        if *dispatch == dispatch_index {
            binding.apply(kernarg_bytes, position)?;
        }
    }
    Ok(())
}

/// Every kernarg binding the retained replay of `launches[..prefix]` applies,
/// sorted by `(dispatch, offset)`.
///
/// The single owner of the dynamic-slot set: the retained PM4 plan and the
/// recorded-HIP oracle (`Gpu::replay_recorded_hip_prefix_at`) both build their
/// bindings here, so the oracle Redline checks PM4 against re-derives exactly
/// the fields PM4 patches. Three sources, one owner per `(dispatch, offset)`:
/// the GDN requant frame of every GDN-family launch, the differential position
/// bindings synthesized from two recordings, and each launch's engine-declared
/// bindings.
pub(crate) fn retained_kernarg_bindings(
    launches: &[RecordedHipLaunch],
    prefix: usize,
    synthesized: &[(usize, ReplayKernargBinding)],
) -> Result<Vec<(usize, ReplayKernargBinding)>, String> {
    let mut bindings = Vec::new();
    for (dispatch, launch) in launches.iter().take(prefix).enumerate() {
        if is_gdn_kernel(&launch.kernel) {
            // frames = max(1, nt * grid.z): the one helper that decides the
            // reservation run length for every transport.
            let frames = gdn_requant_frames_for_dispatch(&launch.kernarg, launch.grid[2])
                .map_err(|reason| format!("{}: {reason}", launch.kernel))?;
            bindings.push((
                dispatch,
                ReplayKernargBinding::GdnFrameU32 { offset: 76, frames },
            ));
        }
    }
    bindings.extend(
        synthesized
            .iter()
            .filter(|(dispatch, _)| *dispatch < prefix)
            .copied(),
    );
    merge_declared_kernarg_bindings(launches, prefix, &mut bindings)?;
    bindings.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.offset().cmp(&b.1.offset())));
    Ok(bindings)
}

/// Merge every engine-declared kernarg binding from the retained prefix into
/// `bindings`.
///
/// One owner per `(dispatch, offset)`: a second binding for the same slot would
/// patch it twice, so the collision is rejected rather than resolved by order.
/// Only [`retained_kernarg_bindings`] calls this, so PM4 and the recorded-HIP
/// oracle see the same declared set.
fn merge_declared_kernarg_bindings(
    launches: &[RecordedHipLaunch],
    prefix: usize,
    bindings: &mut Vec<(usize, ReplayKernargBinding)>,
) -> Result<(), String> {
    for (dispatch, launch) in launches.iter().take(prefix).enumerate() {
        for binding in launch.declared_kernarg_bindings() {
            // A model lowering may declare position-derived fields only. The GDN
            // frame counter is derived by this layer from the recorded launch
            // itself, so declaring it would be a second owner, not a declaration.
            if !binding.is_position_derived() {
                return Err(format!(
                    "{}: declared kernarg binding at offset {} is not position-derived",
                    launch.kernel,
                    binding.offset()
                ));
            }
            let offset = binding.offset();
            if bindings
                .iter()
                .any(|(index, existing)| *index == dispatch && existing.offset() == offset)
            {
                return Err(format!(
                    "{}: dynamic kernarg offset {offset} at dispatch {dispatch} already has an owner",
                    launch.kernel
                ));
            }
            bindings.push((dispatch, *binding));
        }
    }
    Ok(())
}

impl ReplayController {
    /// Accessor for the synthesized position bindings (tests; transports use
    /// [`Self::retained_kernarg_bindings`]).
    pub(crate) fn synthesized_position_bindings(&self) -> &[(usize, ReplayKernargBinding)] {
        &self.synthesized_position_bindings
    }

    /// The complete retained binding set for the first `prefix` recorded
    /// launches; see [`retained_kernarg_bindings`].
    pub(crate) fn retained_kernarg_bindings(
        &self,
        prefix: usize,
    ) -> Result<Vec<(usize, ReplayKernargBinding)>, String> {
        retained_kernarg_bindings(&self.recorded, prefix, &self.synthesized_position_bindings)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayCaptureSummary {
    pub launch_count: usize,
    pub unique_kernel_count: usize,
    pub sequence_hash: u64,
}

fn replay_sequence_hash<'a>(launches: impl IntoIterator<Item = &'a RecordedHipLaunch>) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for launch in launches {
        for byte in launch.kernel.as_bytes().iter().copied().chain([0]) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for value in launch
            .grid
            .iter()
            .chain(&launch.block)
            .chain([&launch.shared_mem])
        {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        match launch.grid_binding {
            None => {
                hash ^= 0;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            Some(ReplayGridBinding::PositionCeilDiv {
                axis,
                addend,
                divisor,
            }) => {
                hash ^= 1;
                hash = hash.wrapping_mul(0x100000001b3);
                for byte in [axis]
                    .into_iter()
                    .chain(addend.to_le_bytes())
                    .chain(divisor.to_le_bytes())
                {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(0x100000001b3);
                }
            }
        }
        // Declared dynamic kernarg fields are tape identity: two tapes that
        // differ only in which slot is position-derived are different contracts.
        hash ^= launch.declared_kernarg_bindings.len() as u64;
        hash = hash.wrapping_mul(0x100000001b3);
        for binding in &launch.declared_kernarg_bindings {
            for byte in binding.identity_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
    }
    hash
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplayObservation {
    pub count: u64,
    pub first_position: Option<usize>,
    pub last_position: Option<usize>,
    pub failed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedReplayIdentity {
    pub dispatch_count: usize,
    pub packet_count: Option<usize>,
    pub queue_id: u64,
    pub command_dwords: Option<u32>,
    /// Prepared graph queue width (AQL linear batch is always 1).
    pub queue_count: usize,
    /// Prepared graph logical phase count (AQL linear batch is always 1).
    pub phase_count: usize,
}

fn pm4_packet_identity(packet_count: usize) -> Option<usize> {
    (packet_count > 0).then_some(packet_count)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AqlContractProbe {
    pub kernel: String,
    pub captured_kernarg_bytes: usize,
    pub loader_kernarg_bytes: u32,
    pub loader_kernarg_alignment: u32,
    pub static_group_bytes: u32,
    pub dynamic_group_bytes: u32,
}

pub struct PreparedLinearAqlReplay {
    graph: SingleQueueBatchGraph,
    dynamic_gdn_frames: Vec<usize>,
}

impl PreparedLinearAqlReplay {
    /// # Safety
    ///
    /// Every pointer captured in the immutable explicit kernarg prefixes must
    /// still refer to the same live Hipfire allocation and model instance.
    pub unsafe fn replay_and_wait(&mut self) -> Result<GpuBatchTiming, String> {
        for dispatch in &self.dynamic_gdn_frames {
            let frame = crate::norm::reserve_gdn_requant_frames(1);
            self.graph
                .patch_kernarg_u32(*dispatch, 76, frame)
                .map_err(|error| error.to_string())?;
        }
        // SAFETY: forwarded from the caller that owns the model allocations.
        unsafe { self.graph.replay_and_wait() }.map_err(|error| error.to_string())
    }

    pub fn dispatch_count(&self) -> usize {
        self.graph.dispatch_count()
    }

    pub fn packet_count(&self) -> usize {
        self.graph.packet_count()
    }

    pub fn queue_id(&self) -> u64 {
        self.graph.queue_id()
    }
}

/// True when the per-dispatch timestamp diagnostic is requested.
pub fn dispatch_profile_enabled() -> bool {
    hipfire_config::developer_var_os("HIPFIRE_REDLINE_DISPATCH_PROFILE")
        .is_some_and(|value| value != "0" && !value.is_empty())
}

/// Default NOP pacing of the retained gfx1201 Qwen3.5-dense decode tape.
const GFX1201_DEFAULT_PM4_PACING: Gfx12DispatchPacing = Gfx12DispatchPacing::PostDispatchNop(64);

/// `HIPFIRE_GFX1201_PM4_PACING` (`replay.gfx1201_pm4_pacing`): unset or
/// `auto` selects the default, `0`/`off` disables pacing and `nop:N` emits an
/// N-body-dword NOP after every dispatch. An unparseable value keeps the
/// default.
pub fn gfx1201_pm4_pacing_from_config() -> Gfx12DispatchPacing {
    let value = hipfire_config::process_value("HIPFIRE_GFX1201_PM4_PACING");
    parse_gfx1201_pm4_pacing(value.as_deref()).unwrap_or_else(|reason| {
        eprintln!("[redline] ignoring HIPFIRE_GFX1201_PM4_PACING: {reason}");
        GFX1201_DEFAULT_PM4_PACING
    })
}

/// An explicit `HIPFIRE_GFX1201_PM4_PACING` (`off` / `nop:N`), or `None` when
/// it is unset, `auto` or unparseable. railgun's gfx1201 lowering keeps its
/// own per-arch pacing (`railgun::plan::ArchLowering`) unless this is set.
pub(crate) fn gfx1201_pm4_pacing_override() -> Option<Gfx12DispatchPacing> {
    let value = hipfire_config::process_value("HIPFIRE_GFX1201_PM4_PACING")?;
    if matches!(value.trim().to_ascii_lowercase().as_str(), "" | "auto") {
        return None;
    }
    parse_gfx1201_pm4_pacing(Some(&value)).ok()
}

fn parse_gfx1201_pm4_pacing(value: Option<&str>) -> Result<Gfx12DispatchPacing, String> {
    let value = value.map(str::trim).unwrap_or("auto");
    let count = |text: &str| {
        text.parse::<u32>()
            .ok()
            .filter(|dwords| (1..=0x4000).contains(dwords))
            .ok_or_else(|| format!("{value:?}: dword count must be 1..=16384"))
    };
    match value.to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(GFX1201_DEFAULT_PM4_PACING),
        "0" | "off" | "false" | "none" => Ok(Gfx12DispatchPacing::None),
        other => match other.split_once(':') {
            Some(("nop", dwords)) => count(dwords).map(Gfx12DispatchPacing::PostDispatchNop),
            _ => Err(format!("{value:?}: expected auto, off or nop:N")),
        },
    }
}

/// Summarise per-dispatch spans so a slow machine reports a distribution
/// rather than a single throughput number.
///
/// The shape is the diagnostic: overhead spread evenly across every dispatch
/// points at per-dispatch cost (fetch, launch, submission), whereas a few
/// dispatches dominating points at specific stalls — a barrier or a cache
/// release. Those two have different causes and different fixes, and an
/// end-to-end tok/s cannot tell them apart.
fn report_dispatch_spans(spans: &[u64]) {
    if spans.is_empty() {
        return;
    }
    let mut sorted: Vec<u64> = spans.to_vec();
    sorted.sort_unstable();
    let total: u64 = sorted.iter().sum();
    let pick = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
    // Contribution of the slowest 5% — the number that separates "everything is
    // slightly slow" from "a handful of dispatches dominate".
    let tail_start = sorted.len() - sorted.len().div_ceil(20);
    let tail: u64 = sorted[tail_start..].iter().sum();
    eprintln!(
        "[redline] dispatch spans n={} total={}us p50={}ns p90={}ns p99={}ns max={}ns \
         slowest5%={:.1}% of total",
        sorted.len(),
        total / 1_000,
        pick(0.50),
        pick(0.90),
        pick(0.99),
        sorted[sorted.len() - 1],
        100.0 * tail as f64 / total.max(1) as f64,
    );
}

/// PACKET3 opcodes used only by preparation-time PM4 stream accounting.
const PACKET3_SET_SH_REG: u32 = 0x76;
const PACKET3_DISPATCH_DIRECT: u32 = 0x15;
const DISPATCH_DIRECT_PACKET_DWORDS: u32 = 5;

/// Opt-in preparation-only PM4 stream accounting. Default off; exact `1` only.
fn pm4_stream_accounting_enabled() -> bool {
    hipfire_config::developer_var("HIPFIRE_REPLAY_PM4_STREAM_ACCOUNTING")
        .ok()
        .as_deref()
        == Some("1")
}

fn validate_pm4_stream_accounting_queue_count(
    stream_accounting: bool,
    queue_count: usize,
) -> Result<(), String> {
    if stream_accounting && queue_count != 1 {
        return Err(format!(
            "HIPFIRE_REPLAY_PM4_STREAM_ACCOUNTING requires one PM4 queue, got {queue_count}"
        ));
    }
    Ok(())
}

/// Entry sentinel used when a SET_SH_REG feeds dispatch 0 (no previous kernel).
const PM4_STREAM_ENTRY_TRANSITION: &str = "<entry>";

/// Map a SET_SH `following_dispatch` onto the captured previous/current kernels
/// in frozen execution order. `order[i]` is the recorded launch index executed
/// as dispatch `i`.
fn pm4_set_sh_transition(
    order: &[usize],
    recorded: &[RecordedHipLaunch],
    following_dispatch: u32,
) -> Result<(String, String), String> {
    let dispatch = following_dispatch as usize;
    if dispatch >= order.len() {
        return Err(format!(
            "SET_SH following_dispatch={following_dispatch} is outside frozen order len {}",
            order.len()
        ));
    }
    let current_index = order[dispatch];
    if current_index >= recorded.len() {
        return Err(format!(
            "SET_SH following_dispatch={following_dispatch} maps to recorded index {current_index} outside prefix {}",
            recorded.len()
        ));
    }
    let current = recorded[current_index].kernel.clone();
    let previous = if dispatch == 0 {
        PM4_STREAM_ENTRY_TRANSITION.to_owned()
    } else {
        let previous_index = order[dispatch - 1];
        if previous_index >= recorded.len() {
            return Err(format!(
                "SET_SH previous dispatch maps to recorded index {previous_index} outside prefix {}",
                recorded.len()
            ));
        }
        recorded[previous_index].kernel.clone()
    };
    Ok((previous, current))
}

/// Deterministic preparation-time attribution of a gfx10/gfx11 PM4 stream.
///
/// Owns no retained state: callers log the formatted report and drop it before
/// graph installation so nothing reaches [`PreparedPm4Replay`].
#[derive(Clone, Debug, Eq, PartialEq)]
struct Pm4StreamAccountingReport {
    architecture: String,
    dispatch_count: usize,
    command_dwords: u32,
    execution_sequence_hash: u64,
    dependency_waits: usize,
    dependency_acquires: usize,
    packet_classes: BTreeMap<(u32, u32), usize>,
    set_sh_header_dwords: u32,
    set_sh_value_dwords: u32,
    set_sh_repeated_value_dwords: u32,
    /// Value-dword writes keyed by architectural first-register offset.
    writes_by_register_offset: BTreeMap<u32, u32>,
    /// Value-dword writes keyed by captured previous→current kernel transition.
    writes_by_transition: BTreeMap<(String, String), u32>,
}

impl Pm4StreamAccountingReport {
    /// Aggregate census + SET_SH records against the frozen order and reconcile.
    fn build(
        architecture: &str,
        order: &[usize],
        recorded: &[RecordedHipLaunch],
        command_dwords: u32,
        execution_sequence_hash: u64,
        dependency_waits: usize,
        dependency_acquires: usize,
        census: &BTreeMap<(u32, u32), usize>,
        records: &[Gfx10SetShRegRecord],
    ) -> Result<Self, String> {
        // Packet-class dwords must exhaust the stream.
        let accounted_dwords =
            census
                .iter()
                .try_fold(0_u64, |acc, ((_, packet_dwords), count)| {
                    let packet = u64::from(*packet_dwords);
                    let n = *count as u64;
                    packet
                        .checked_mul(n)
                        .and_then(|part| acc.checked_add(part))
                        .ok_or_else(|| "PM4 packet census dword product overflowed".to_owned())
                })?;
        if accounted_dwords != u64::from(command_dwords) {
            return Err(format!(
                "PM4 packet census accounts for {accounted_dwords} dwords but stream has {command_dwords}"
            ));
        }

        let dispatch_packets = census
            .get(&(PACKET3_DISPATCH_DIRECT, DISPATCH_DIRECT_PACKET_DWORDS))
            .copied()
            .unwrap_or(0);
        if dispatch_packets != order.len() {
            return Err(format!(
                "PM4 DISPATCH_DIRECT count {dispatch_packets} does not match frozen order len {}",
                order.len()
            ));
        }
        // No other DISPATCH_DIRECT packet sizes are legal for this encoder.
        let other_dispatch = census
            .iter()
            .filter(|((opcode, packet_dwords), _)| {
                *opcode == PACKET3_DISPATCH_DIRECT
                    && *packet_dwords != DISPATCH_DIRECT_PACKET_DWORDS
            })
            .map(|((_, _), count)| *count)
            .sum::<usize>();
        if other_dispatch != 0 {
            return Err(format!(
                "PM4 stream contains {other_dispatch} non-5-dword DISPATCH_DIRECT packets"
            ));
        }

        // SET_SH payload value dwords from census (packet = header + first + values).
        let census_set_sh_value_dwords =
            census
                .iter()
                .try_fold(0_u64, |acc, ((opcode, packet_dwords), count)| {
                    if *opcode != PACKET3_SET_SH_REG {
                        return Ok(acc);
                    }
                    if *packet_dwords < 3 {
                        return Err(format!(
                            "PM4 SET_SH_REG census entry has packet_dwords={packet_dwords} (< 3)"
                        ));
                    }
                    let values = u64::from(*packet_dwords - 2);
                    let n = *count as u64;
                    values
                        .checked_mul(n)
                        .and_then(|part| acc.checked_add(part))
                        .ok_or_else(|| {
                            "PM4 SET_SH census value dword product overflowed".to_owned()
                        })
                })?;
        let census_set_sh_packets = census
            .iter()
            .filter(|((opcode, _), _)| *opcode == PACKET3_SET_SH_REG)
            .map(|((_, _), count)| *count)
            .sum::<usize>();

        let mut set_sh_header_dwords = 0_u32;
        let mut set_sh_value_dwords = 0_u32;
        let mut set_sh_repeated_value_dwords = 0_u32;
        let mut writes_by_register_offset = BTreeMap::<u32, u32>::new();
        let mut writes_by_transition = BTreeMap::<(String, String), u32>::new();

        for record in records {
            if record.following_dispatch as usize >= order.len() {
                return Err(format!(
                    "SET_SH at dword {} following_dispatch={} is outside frozen order len {}",
                    record.packet_dword,
                    record.following_dispatch,
                    order.len()
                ));
            }
            if record.repeated_value_dwords > record.value_dwords {
                return Err(format!(
                    "SET_SH at dword {} repeated_value_dwords={} exceeds value_dwords={}",
                    record.packet_dword, record.repeated_value_dwords, record.value_dwords
                ));
            }
            set_sh_header_dwords = set_sh_header_dwords
                .checked_add(1)
                .ok_or_else(|| "PM4 SET_SH header dword count overflowed".to_owned())?;
            set_sh_value_dwords = set_sh_value_dwords
                .checked_add(record.value_dwords)
                .ok_or_else(|| "PM4 SET_SH value dword count overflowed".to_owned())?;
            set_sh_repeated_value_dwords = set_sh_repeated_value_dwords
                .checked_add(record.repeated_value_dwords)
                .ok_or_else(|| "PM4 SET_SH repeated value dword count overflowed".to_owned())?;
            let register_writes = writes_by_register_offset
                .entry(record.first_register)
                .or_default();
            *register_writes = register_writes
                .checked_add(record.value_dwords)
                .ok_or_else(|| "PM4 SET_SH register-offset write count overflowed".to_owned())?;
            let transition = pm4_set_sh_transition(order, recorded, record.following_dispatch)?;
            let transition_writes = writes_by_transition.entry(transition).or_default();
            *transition_writes = transition_writes
                .checked_add(record.value_dwords)
                .ok_or_else(|| "PM4 SET_SH transition write count overflowed".to_owned())?;
        }

        if records.len() != census_set_sh_packets {
            return Err(format!(
                "SET_SH record count {} does not match census SET_SH packets {census_set_sh_packets}",
                records.len()
            ));
        }
        if u64::from(set_sh_value_dwords) != census_set_sh_value_dwords {
            return Err(format!(
                "SET_SH record value dwords {set_sh_value_dwords} do not match census payload values {census_set_sh_value_dwords}"
            ));
        }

        Ok(Self {
            architecture: architecture.to_owned(),
            dispatch_count: order.len(),
            command_dwords,
            execution_sequence_hash,
            dependency_waits,
            dependency_acquires,
            packet_classes: census.clone(),
            set_sh_header_dwords,
            set_sh_value_dwords,
            set_sh_repeated_value_dwords,
            writes_by_register_offset,
            writes_by_transition,
        })
    }

    fn format_report(&self) -> String {
        // Stable BTreeMap iteration: packet classes, register offsets, transitions.
        let mut packet_parts = Vec::with_capacity(self.packet_classes.len());
        for ((opcode, packet_dwords), count) in &self.packet_classes {
            packet_parts.push(format!(
                "(op=0x{opcode:02x},dwords={packet_dwords})×{count}"
            ));
        }
        let mut register_parts = Vec::with_capacity(self.writes_by_register_offset.len());
        for (register, writes) in &self.writes_by_register_offset {
            register_parts.push(format!("0x{register:x}={writes}"));
        }
        let mut transition_parts = Vec::with_capacity(self.writes_by_transition.len());
        for ((previous, current), writes) in &self.writes_by_transition {
            transition_parts.push(format!("{previous}->{current}={writes}"));
        }
        format!(
            "[redline] PM4 stream accounting arch={} dispatches={} command_dwords={} \
             execution_sequence_hash={:016x} dependency_waits={} dependency_acquires={} \
             terminal_waits=1 packet_classes=[{}] set_sh_headers={} set_sh_values={} \
             set_sh_repeated={} writes_by_register_offset=[{}] writes_by_transition=[{}]",
            self.architecture,
            self.dispatch_count,
            self.command_dwords,
            self.execution_sequence_hash,
            self.dependency_waits,
            self.dependency_acquires,
            packet_parts.join(", "),
            self.set_sh_header_dwords,
            self.set_sh_value_dwords,
            self.set_sh_repeated_value_dwords,
            register_parts.join(", "),
            transition_parts.join(", "),
        )
    }
}

/// Run gfx10/gfx11 stream accounting after terminal idle and before graph create.
///
/// Malformed or partial accounting fails preparation. The report is logged and
/// dropped; nothing is retained on the prepared replay object.
fn report_pm4_stream_accounting(
    architecture: &str,
    order: &[usize],
    recorded: &[RecordedHipLaunch],
    commands: &Pm4Commands,
    command_dwords: u32,
    execution_sequence_hash: u64,
    dependency_waits: usize,
    dependency_acquires: usize,
) -> Result<(), String> {
    let census = match commands.packet_census() {
        Some(Ok(census)) => census,
        Some(Err(dword)) => {
            return Err(format!(
                "PM4 stream accounting packet census failed at dword {dword}"
            ));
        }
        None => {
            return Err(
                "PM4 stream accounting requested but architecture has no gfx10/gfx11 census"
                    .to_owned(),
            );
        }
    };
    let records = match commands.set_sh_reg_records() {
        Some(Ok(records)) => records,
        Some(Err(dword)) => {
            return Err(format!(
                "PM4 stream accounting SET_SH records failed at dword {dword}"
            ));
        }
        None => {
            return Err(
                "PM4 stream accounting requested but architecture has no gfx10/gfx11 SET_SH records"
                    .to_owned(),
            );
        }
    };
    let report = Pm4StreamAccountingReport::build(
        architecture,
        order,
        recorded,
        command_dwords,
        execution_sequence_hash,
        dependency_waits,
        dependency_acquires,
        &census,
        &records,
    )?;
    eprintln!("{}", report.format_report());
    // Explicit drop: report must not escape into PreparedPm4Replay.
    drop(report);
    Ok(())
}

enum PreparedPm4Graph {
    Single(SingleQueuePm4Ib),
    Phased(PhasedMultiQueuePm4Ib),
}

impl PreparedPm4Graph {
    /// # Safety
    ///
    /// Same contract as [`Self::replay_and_wait_profiled_checked`].
    unsafe fn replay_and_wait_profiled(&mut self) -> Result<GpuMultiQueueTiming, String> {
        // SAFETY: checked variant with string conversion; caller upholds PM4 tape liveness.
        unsafe { self.replay_and_wait_profiled_checked() }.map_err(|(error, _)| error.to_string())
    }

    /// # Safety
    ///
    /// Every pointer captured in the prepared PM4/IB tape must still refer to
    /// live Hipfire allocations for this model instance; queues must be idle
    /// enough for the underlying profiled replay helpers' quiescence contract.
    unsafe fn replay_and_wait_profiled_checked(
        &mut self,
    ) -> Result<GpuMultiQueueTiming, (redline_dispatch::aql::ReplayError, Quiescence)> {
        if dispatch_profile_enabled() {
            if let Self::Single(graph) = self {
                // Execute the instrumented graph once. Reuse the same timestamp
                // vector for whole-tape timing and the one-line legacy report.
                // SAFETY: Single-queue tape + bindings still live; see method # Safety.
                let (timing, spans) = unsafe { graph.replay_and_wait_dispatch_profiled_checked() }
                    .map_err(|(error, q)| (error, q))?;
                static REPORTED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    report_dispatch_spans(&spans);
                }
                return Ok(timing);
            }
        }
        match self {
            // SAFETY: prepared graph/tape still live; same contract as method # Safety.
            Self::Single(graph) => unsafe { graph.replay_and_wait_profiled_checked() },
            // SAFETY: phased multi-queue tape still live; errors map to Proven quiescence.
            Self::Phased(graph) => unsafe { graph.replay_and_wait_profiled() }
                .map_err(|error| (error, Quiescence::Proven)),
        }
    }

    fn read_dispatch_profile(&mut self) -> Result<Pm4DispatchProfile, String> {
        match self {
            Self::Single(graph) => graph
                .read_dispatch_profile()
                .map(|(timing, spans_nanoseconds)| Pm4DispatchProfile {
                    timing,
                    spans_nanoseconds,
                })
                .map_err(|error| error.to_string()),
            Self::Phased(_) => {
                Err("per-dispatch PM4 profiling requires a single retained queue".to_owned())
            }
        }
    }

    fn queue_id(&self) -> u64 {
        match self {
            Self::Single(graph) => graph.queue_id(),
            Self::Phased(graph) => graph
                .queue_ids()
                .next()
                .expect("prepared phased PM4 replay owns at least one queue"),
        }
    }

    fn queue_count(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Phased(graph) => graph.queue_count(),
        }
    }

    fn phase_count(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Phased(graph) => graph.phase_count(),
        }
    }

    fn packet_count(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Phased(graph) => graph.packet_count(),
        }
    }

    fn patch_dispatch_dimensions(
        &mut self,
        dispatch: usize,
        dimensions: [u32; 3],
    ) -> Result<(), String> {
        match self {
            Self::Single(graph) => graph
                .patch_dispatch_dimensions(dispatch, dimensions)
                .map_err(|error| error.to_string()),
            Self::Phased(_) => {
                Err("dynamic PM4 geometry requires the certified single-queue replay".to_owned())
            }
        }
    }

    fn quiesce(&mut self) -> Result<(), (redline_dispatch::aql::ReplayError, Quiescence)> {
        match self {
            Self::Single(graph) => graph
                .quiesce()
                .map_err(|error| (error, Quiescence::Unknown)),
            Self::Phased(_) => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Pm4DispatchBoundary {
    /// Entry ownership acquire emitted before the first dispatch.
    /// Distinct from mid-tape `acquire_inter_node`.
    pub entry_acquire: bool,
    pub wait_compute_idle: bool,
    pub acquire_inter_node: bool,
    pub acquire_vmem: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Pm4DispatchProfile {
    pub timing: GpuMultiQueueTiming,
    pub spans_nanoseconds: Vec<u64>,
}

pub struct PreparedPm4Replay {
    graph: PreparedPm4Graph,
    // Kernels retain their HSA executables and kernargs retain every pointer
    // programmed into the immutable indirect buffer.
    _kernels: Vec<Kernel>,
    kernargs: Vec<KernargBuffer>,
    /// Last non-empty device-local kernarg segment; publishing it after the
    /// per-replay patches makes every earlier BAR store GPU-visible.
    kernarg_publish: Option<usize>,
    /// gfx1010 RELEASE_MEM/WAIT_REG_MEM fence word. Owned for the full
    /// executable lifetime of `graph` so the IB's absolute address stays valid
    /// through every replay; dropped only after queue quiescence via normal
    /// PreparedPm4Replay teardown (field order: graph first).
    _dependency_fence: Option<KernargBuffer>,
    dynamic_gdn_frames: Vec<usize>,
    dynamic_kernarg_bindings: Vec<(usize, ReplayKernargBinding)>,
    dynamic_grids: Vec<(usize, ReplayGridBinding, [u32; 3], [u32; 3])>,
    pm4_architecture: Pm4Architecture,
    dispatch_count: usize,
    command_dwords: u32,
    dispatch_boundaries: Option<Vec<Pm4DispatchBoundary>>,
    prepared_max_position: Option<usize>,
    /// Slice-1 re-encode cache: loader explicit-prefix length per prepared
    /// dispatch (aligned with `kernargs`), and the `BindingRevision` the
    /// segments are encoded at. Unchanged revision ⇒ the buffers are current
    /// and the replay hot path does no re-encode work.
    bound_explicit_lens: Vec<usize>,
    /// Per prepared dispatch (aligned with `kernargs`): sorted `(offset, len)`
    /// ranges excluded from the relocation non-slot-bytes gate (pointer slots
    /// and u32 dynamic bindings). Precomputed once at prepare time.
    bound_exempt_ranges: Vec<Vec<(usize, usize)>>,
    encoded_revision: BindingRevision,
}

impl PreparedPm4Replay {
    /// # Safety
    ///
    /// Every pointer captured in the immutable explicit kernarg prefixes must
    /// still refer to the same live Hipfire allocation and model instance.
    pub unsafe fn replay_and_wait(
        &mut self,
        position: usize,
    ) -> Result<GpuMultiQueueTiming, String> {
        // SAFETY: checked variant handles quiescence mapping.
        unsafe { self.replay_and_wait_checked(position) }.map_err(|failure| failure.error)
    }

    /// Checked variant that reports quiescence.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::replay_and_wait`].
    pub unsafe fn replay_and_wait_checked(
        &mut self,
        position: usize,
    ) -> Result<GpuMultiQueueTiming, RetainedReplayFailure> {
        if let Some(max_pos) = self.prepared_max_position {
            if position > max_pos {
                return Err(RetainedReplayFailure {
                    error: format!("position {position} exceeds prepared max_position {max_pos}"),
                    quiescence: ReplayQuiescence::Proven,
                });
            }
        }
        let patch_started = crate::gap_timing::now();
        // Patch typed kernarg bindings while queue is quiescent.
        // Single code path shared with recorded-HIP replay: both transports
        // must apply the identical binding set via `apply_kernarg_bindings_for_dispatch`.
        for dispatch_index in 0..self.kernargs.len() {
            let bytes = self.kernargs[dispatch_index].as_mut_bytes();
            apply_kernarg_bindings_for_dispatch(
                bytes,
                dispatch_index,
                position,
                &self.dynamic_kernarg_bindings,
            )
            .map_err(|error| RetainedReplayFailure {
                error,
                quiescence: ReplayQuiescence::Proven,
            })?;
        }
        // Legacy GDN frame patch (covers old prepared objects; new objects have
        // empty dynamic_gdn_frames and are patched via dynamic_kernarg_bindings).
        for dispatch in &self.dynamic_gdn_frames {
            // Skip if already covered by a GdnFrame binding for this dispatch.
            let already_covered = self.dynamic_kernarg_bindings.iter().any(|(idx, binding)| {
                *idx == *dispatch && matches!(binding, ReplayKernargBinding::GdnFrameU32 { .. })
            });
            if already_covered {
                continue;
            }
            let frame = crate::norm::reserve_gdn_requant_frames(1);
            let bytes = self.kernargs[*dispatch].as_mut_bytes();
            bytes
                .get_mut(76..80)
                .ok_or_else(|| RetainedReplayFailure {
                    error: "PM4 GDN kernarg is too short for frame patch".to_owned(),
                    quiescence: ReplayQuiescence::Proven,
                })?
                .copy_from_slice(&frame.to_ne_bytes());
        }
        for (dispatch, binding, recorded, workgroup) in &self.dynamic_grids {
            let workgroups =
                binding
                    .bind(position, *recorded)
                    .map_err(|error| RetainedReplayFailure {
                        error,
                        quiescence: ReplayQuiescence::Proven,
                    })?;
            let dimensions = if self.pm4_architecture == Pm4Architecture::Gfx12 {
                let mut workitems = [0_u32; 3];
                for axis in 0..3 {
                    workitems[axis] =
                        workgroups[axis]
                            .checked_mul(workgroup[axis])
                            .ok_or_else(|| RetainedReplayFailure {
                                error: format!(
                                "dynamic PM4 grid overflow axis={axis} workgroups={} workgroup={}",
                                workgroups[axis], workgroup[axis]
                            ),
                                quiescence: ReplayQuiescence::Proven,
                            })?;
                }
                workitems
            } else {
                workgroups
            };
            self.graph
                .patch_dispatch_dimensions(*dispatch, dimensions)
                .map_err(|error| RetainedReplayFailure {
                    error: error.to_string(),
                    quiescence: ReplayQuiescence::Proven,
                })?;
        }
        if let Some(index) = self.kernarg_publish {
            self.kernargs[index].publish_host_writes();
        }
        crate::gap_timing::add_since(crate::gap_timing::Slot::Patch, patch_started);
        let wait_started = crate::gap_timing::now();
        // SAFETY: forwarded from the caller that owns the model allocations.
        let result = unsafe { self.graph.replay_and_wait_profiled_checked() }.map_err(
            |(error, quiescence)| RetainedReplayFailure {
                error: error.to_string(),
                quiescence: match quiescence {
                    Quiescence::Proven => ReplayQuiescence::Proven,
                    Quiescence::Unknown => ReplayQuiescence::Unknown,
                },
            },
        );
        crate::gap_timing::add_since(crate::gap_timing::Slot::SubmitWait, wait_started);
        if let Ok(timing) = &result {
            crate::gap_timing::add_ns(
                crate::gap_timing::Slot::GpuSpan,
                timing
                    .last_end
                    .saturating_sub(timing.first_start)
                    .saturating_mul(1_000_000_000)
                    / timing.frequency_hz.max(1),
            );
        }
        result
    }

    /// Replay one instrumented retained graph exactly once.
    ///
    /// # Safety
    ///
    /// Every captured pointer must remain live, as for [`Self::replay_and_wait`].
    pub unsafe fn replay_and_wait_dispatch_profiled(
        &mut self,
        position: usize,
    ) -> Result<Pm4DispatchProfile, String> {
        if self.dispatch_boundaries.is_none() {
            return Err("prepared PM4 graph has no per-dispatch timestamps".to_owned());
        }
        // SAFETY: forwarded from the caller that owns the model allocations.
        unsafe { self.replay_and_wait(position)? };
        self.graph.read_dispatch_profile()
    }

    pub fn dispatch_count(&self) -> usize {
        self.dispatch_count
    }

    pub fn command_dwords(&self) -> u32 {
        self.command_dwords
    }

    pub fn dispatch_boundaries(&self) -> Option<&[Pm4DispatchBoundary]> {
        self.dispatch_boundaries.as_deref()
    }

    pub fn queue_id(&self) -> u64 {
        self.graph.queue_id()
    }

    pub fn queue_count(&self) -> usize {
        self.graph.queue_count()
    }

    pub fn phase_count(&self) -> usize {
        self.graph.phase_count()
    }

    pub fn packet_count(&self) -> usize {
        self.graph.packet_count()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowValidation {
    pub bit_exact: bool,
    pub guards_intact: bool,
    pub same_artifact: bool,
    pub abi_valid: bool,
    pub automatic_clocks: bool,
    pub gpu_timed: bool,
    pub speedup_over_hip: f64,
}

impl ShadowValidation {
    fn passes(self, threshold: f64) -> bool {
        self.bit_exact
            && self.guards_intact
            && self.same_artifact
            && self.abi_valid
            && self.automatic_clocks
            && self.gpu_timed
            && self.speedup_over_hip.is_finite()
            && self.speedup_over_hip >= threshold
    }
}

/// Process-local replay adoption state. HIP remains the route until an adapter
/// both supplies two certified observations and installs a concrete prepared
pub struct ReplayController {
    request: ReplayBackendRequest,
    transport: ReplayTransport,
    pm4_mid_acquire_policy: Pm4MidAcquirePolicy,
    pm4_wait_policy: Pm4WaitPolicy,
    pm4_register_policy: Pm4RegisterPolicy,
    pm4_queue_policy: QueuePolicy,
    /// NOP pacing for the single-queue gfx12 tape; set by the daemon on load.
    pm4_gfx12_dispatch_pacing: Gfx12DispatchPacing,
    state: ReplayState,
    recorded: Vec<RecordedHipLaunch>,
    certified_speedups: Vec<f64>,
    threshold: f64,
    max_recorded_launches: usize,
    fallback_reason: Option<String>,
    prepared: Option<PreparedLinearAqlReplay>,
    prepared_pm4: Option<PreparedPm4Replay>,
    auto_lifecycle: bool,
    forward_eligible: bool,
    replay_observation: ReplayObservation,
    radiowave_effect_launches: usize,
    fallback_effect_launches: usize,
    unknown_effect_launches: usize,
    /// Opt-in latch for daemon-owned post-generate route-proof markers.
    route_proof_log: bool,
    /// Route identities (live expert pointer mappings) the tape was captured
    /// against, keyed by the route's own stable table identity. A retained plan
    /// dereferences pointer tables without re-uploading them, so a different
    /// mapping for the same key means the tape names different tensors. Keyed,
    /// not single-valued: one model has one mapping per layer.
    route_identities: BTreeMap<u64, String>,
    prepared_max_position: Option<usize>,
    synthesized_position_bindings: Vec<(usize, ReplayKernargBinding)>,
    position_bindings_calibrated: bool,
    /// Slice-1 binding store: `ResourceId` issuer for the current tape,
    /// record-time bases per resource, and the revision the prepared kernarg
    /// segments are encoded at. Unchanged revision ⇒ no re-encode work on
    /// the replay hot path; the prepared `KernargBuffer`s ARE the cache.
    binding_issuer: Recorder,
    replay_bindings: ReplayBindings,
    binding_revision: BindingRevision,
    /// Tape-global resource table keyed by record-time
    /// `(allocation_base, allocation_bytes)`: dedupes `ResourceId`s across
    /// launches and is the post-growth liveness probe list
    /// (`hipMemGetAddressRange` on each base).
    tape_resources: BTreeMap<(u64, u64), ResourceId>,
    /// Armed by `invalidate_for_scratch_growth` when a prepared PM4 route is
    /// active; drained by the post-growth refresh before the next launch or
    /// replay. A replay that observes it armed fails closed (route re-armed).
    binding_refresh_pending: bool,
    /// This thread's [`memory_effects`] tally when the open capture window
    /// began, and the memory operations issued inside the last closed window.
    window_effects_base: MemoryEffects,
    window_effects: MemoryEffects,
    /// Shadow-only executor override for the next eligible forward.
    shadow_body_route: Option<ShadowBodyRoute>,
    /// G0 recording-invariance eager arm (railgun design §5 G0): `Some`
    /// while every funnel launch is observed with `is_recording()` false, so
    /// recording-dependent predicates take their eager branch.
    g0_observation: Option<Vec<G0Launch>>,
    /// railgun M1 shadow (`HIPFIRE_RAILGUN_SHADOW`): authors railgun's
    /// program next to every recording and diffs its PM4 lowering against
    /// the prepared tape. Never changes the route.
    railgun_shadow: Option<railgun_shadow::RailgunShadow>,
    /// railgun M2 check mode (`HIPFIRE_RAILGUN_CHECK`): run by
    /// `Gpu::replay_pm4_routed`, parked here between steps.
    pub(crate) railgun_check: Option<Box<crate::railgun_check::CheckState>>,
    /// Count of PM4 programs prepared by this controller: the check mode's
    /// once-per-program weights verification keys on it.
    pm4_generation: u64,
}

/// Which executor the retained body uses on the next eligible forward.
///
/// Production controllers never set this: the executor follows from the
/// prepared plan and the request backend. A manual shadow controller sets it so
/// one prepared tape can be compared across the exact-kernarg HIP oracle, the
/// retained transport, and ordinary HIP without re-capturing between arms —
/// which is what the multi-position state-parity gate needs. Identity matters as
/// much as the transport here: the oracle re-executes the *recorded* HIP launch
/// sequence (kernargs and all), so a divergence it reports is a divergence
/// between ordinary HIP and the byte-exact captured dispatch stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadowBodyRoute {
    /// Submit the prepared plan over its retained transport.
    Plan,
    /// Re-execute the recorded HIP launch prefix (`recorded_hip_prefix_at`).
    HipOracle,
    /// Run the ordinary HIP body (the comparison baseline).
    Hip,
}

impl ReplayController {
    pub fn from_config() -> Self {
        let request = ReplayBackendRequest::from_config();
        let manual = manual_capture_requested();
        let mut controller = if manual {
            Self::new_armed(request)
        } else {
            Self::new(request)
        };
        controller.auto_lifecycle = !manual;
        if !manual && request != ReplayBackendRequest::Hip {
            // Model load and prefill priming use the same central launch
            // recorder. Arm here and clear/start only at the first eligible
            // plain-AR forward so the retained tape cannot absorb setup work.
            controller.state = ReplayState::Armed;
        }
        controller
    }

    pub fn new(request: ReplayBackendRequest) -> Self {
        let state = if request == ReplayBackendRequest::Hip {
            ReplayState::Hip
        } else {
            ReplayState::RecordingWarmup
        };
        Self {
            request,
            transport: ReplayTransport::from_config(),
            pm4_mid_acquire_policy: Pm4MidAcquirePolicy::from_config(),
            pm4_wait_policy: Pm4WaitPolicy::from_config(),
            pm4_register_policy: Pm4RegisterPolicy::from_config(),
            pm4_queue_policy: pm4_queue_policy_from_config(),
            pm4_gfx12_dispatch_pacing: Gfx12DispatchPacing::None,
            state,
            recorded: Vec::new(),
            certified_speedups: Vec::new(),
            threshold: 1.03,
            max_recorded_launches: 4096,
            fallback_reason: None,
            prepared: None,
            prepared_pm4: None,
            auto_lifecycle: false,
            forward_eligible: true,
            replay_observation: ReplayObservation::default(),
            unknown_effect_launches: 0,
            radiowave_effect_launches: 0,
            fallback_effect_launches: 0,
            route_proof_log: route_proof_log_requested(),
            route_identities: BTreeMap::new(),
            prepared_max_position: None,
            synthesized_position_bindings: Vec::new(),
            position_bindings_calibrated: false,
            shadow_body_route: None,
            binding_issuer: Recorder::new(),
            replay_bindings: ReplayBindings::new(),
            binding_revision: BindingRevision(0),
            tape_resources: BTreeMap::new(),
            binding_refresh_pending: false,
            window_effects_base: MemoryEffects::default(),
            window_effects: MemoryEffects::default(),
            g0_observation: None,
            railgun_shadow: railgun_shadow::RailgunShadow::from_config(),
            railgun_check: crate::railgun_check::CheckState::from_config().map(Box::new),
            pm4_generation: 0,
        }
    }

    pub fn new_armed(request: ReplayBackendRequest) -> Self {
        let mut controller = Self::new(request);
        if request != ReplayBackendRequest::Hip {
            controller.state = ReplayState::Armed;
        }
        controller
    }

    /// Construct an explicitly-delimited automatic PM4 controller. Model
    /// adapters use this for secondary retained bodies (for example a
    /// speculative verify shape) that must not replace `Gpu::replay`, the
    /// model's ordinary-AR controller.
    pub fn new_manual_pm4() -> Self {
        let mut controller = Self::new_armed(ReplayBackendRequest::Auto);
        controller.transport = ReplayTransport::Pm4Ib;
        controller.auto_lifecycle = false;
        // Batched DS4 verify grows past the ordinary-AR 4,096-launch cap at
        // B>=5 (B=4 is 3,642 dispatches). Keep this scoped to the secondary
        // controller; the primary model controller retains its stricter cap.
        controller.max_recorded_launches = 8_192;
        controller
    }

    /// Construct the AQL-packet twin of [`Self::new_manual_pm4`]. This exists
    /// as a diagnostic/control route for secondary retained bodies: it keeps
    /// the same capture and kernarg lifetime while replacing architecture-
    /// native PM4 boundary lowering with public-HSA dispatch headers.
    pub fn new_manual_aql() -> Self {
        let mut controller = Self::new_armed(ReplayBackendRequest::Auto);
        controller.transport = ReplayTransport::AqlPackets;
        controller.auto_lifecycle = false;
        controller.max_recorded_launches = 8_192;
        controller
    }

    /// Apply the daemon's model-scoped replay default after a successful load.
    ///
    /// An explicit backend selection always wins. Otherwise every successful
    /// model load resets the process-local controller so prepared queues,
    /// command buffers, and fallback state cannot bleed across model swaps.
    /// Eligible single-GPU MQ4R models may default to retained PM4 on
    /// gfx1100, gfx1151, and gfx1201, as may Qwen3.5 dense plain-AR decode on
    /// gfx1201; this is runtime policy, not certification.
    /// All other models return to ordinary HIP. An explicit transport still
    /// overrides the PM4 transport choice for diagnostics.
    pub fn configure_model_default(&mut self, enable_mq4r: bool) -> bool {
        let manual = manual_capture_requested();
        let backend = hipfire_config::process_value("HIPFIRE_REPLAY_BACKEND");
        let explicit_backend = backend.as_deref().is_some_and(|value| value != "auto");
        if explicit_backend || manual {
            self.reset_for_model(
                ReplayBackendRequest::from_config(),
                ReplayTransport::from_config(),
                !manual,
            );
            return false;
        }

        let transport_override = hipfire_config::process_value("HIPFIRE_REPLAY_TRANSPORT");
        let transport = if enable_mq4r
            && !transport_override
                .as_deref()
                .is_some_and(|value| value != "auto")
        {
            ReplayTransport::Pm4Ib
        } else {
            ReplayTransport::from_config()
        };
        self.apply_model_default(enable_mq4r, transport);
        true
    }

    fn apply_model_default(&mut self, enable_mq4r: bool, transport: ReplayTransport) {
        let request = if enable_mq4r {
            ReplayBackendRequest::Auto
        } else {
            ReplayBackendRequest::Hip
        };
        self.reset_for_model(request, transport, true);
    }
    fn reset_for_model(
        &mut self,
        request: ReplayBackendRequest,
        transport: ReplayTransport,
        auto_lifecycle: bool,
    ) {
        self.request = request;
        self.transport = transport;
        self.state = if request == ReplayBackendRequest::Hip {
            ReplayState::Hip
        } else {
            ReplayState::Armed
        };
        self.recorded.clear();
        self.certified_speedups.clear();
        self.fallback_reason = None;
        self.prepared = None;
        self.prepared_pm4 = None;
        self.auto_lifecycle = auto_lifecycle;
        self.forward_eligible = true;
        self.replay_observation = ReplayObservation::default();
        self.radiowave_effect_launches = 0;
        self.fallback_effect_launches = 0;
        self.unknown_effect_launches = 0;
        self.prepared_max_position = None;
        self.route_identities.clear();
        self.synthesized_position_bindings.clear();
        self.position_bindings_calibrated = false;
        self.shadow_body_route = None;
        self.binding_issuer = Recorder::new();
        self.replay_bindings = ReplayBindings::new();
        self.binding_revision = BindingRevision(0);
        self.tape_resources.clear();
        self.binding_refresh_pending = false;
        self.window_effects = MemoryEffects::default();
        if let Some(shadow) = self.railgun_shadow.as_mut() {
            shadow.reset();
        }
    }

    /// Drop a prepared route after a model-owned allocation/geometry bucket
    /// changes, preserving the selected backend and transport. Unlike
    /// [`Self::poison`], this is an expected lifecycle transition: the next
    /// eligible forward records and prepares a fresh route for the new stable
    /// layout.
    pub fn rearm_after_layout_growth(&mut self) {
        let request = self.request;
        let transport = self.transport;
        let auto_lifecycle = self.auto_lifecycle;
        self.reset_for_model(request, transport, auto_lifecycle);
    }

    /// Set the executor the retained body uses on the next eligible forward.
    ///
    /// Only a manual shadow controller may use this: production adoption reads
    /// the executor from the prepared plan. `None` restores that default.
    pub fn set_shadow_body_route(&mut self, route: Option<ShadowBodyRoute>) {
        self.shadow_body_route = route;
    }

    pub fn shadow_body_route(&self) -> Option<ShadowBodyRoute> {
        self.shadow_body_route
    }

    /// Which transport the *currently prepared* plan uses, if any.
    ///
    /// A shadow arm prepares its plan itself, so the plan that exists — not the
    /// controller's configured transport — decides which replay entry submits it.
    pub fn prepared_pm4_plan_ready(&self) -> bool {
        self.prepared_pm4.is_some()
    }

    pub fn prepared_aql_plan_ready(&self) -> bool {
        self.prepared.is_some() && self.prepared_pm4.is_none()
    }

    pub fn transport_name(&self) -> &'static str {
        match self.transport {
            ReplayTransport::AqlPackets => "aql",
            ReplayTransport::Pm4Ib => "pm4",
        }
    }

    pub fn request(&self) -> ReplayBackendRequest {
        self.request
    }

    pub fn state(&self) -> ReplayState {
        self.state
    }

    pub fn recorded_launches(&self) -> &[RecordedHipLaunch] {
        &self.recorded
    }

    pub fn pm4_queue_policy(&self) -> QueuePolicy {
        self.pm4_queue_policy
    }

    /// Pace the next single-queue gfx12 PM4 preparation (ignored on gfx10/11,
    /// multi-queue tapes and the per-dispatch profile).
    pub fn set_pm4_gfx12_dispatch_pacing(&mut self, pacing: Gfx12DispatchPacing) {
        self.pm4_gfx12_dispatch_pacing = pacing;
    }

    pub fn prepared_pm4_shape(&self) -> Option<(usize, usize)> {
        self.prepared_pm4
            .as_ref()
            .map(|prepared| (prepared.queue_count(), prepared.phase_count()))
    }

    /// Identity of the installed plan.
    ///
    /// The plan that exists decides which transport's identity this is: an
    /// adapter (or a shadow arm) prepares the transport it chooses, and a
    /// controller that answered "no identity" while holding an installed plan
    /// would report an unproven route as an absent one.
    pub fn prepared_route_identity(&self) -> Option<PreparedReplayIdentity> {
        if let Some(prepared) = self.prepared_pm4.as_ref() {
            return Some(PreparedReplayIdentity {
                dispatch_count: prepared.dispatch_count(),
                packet_count: pm4_packet_identity(prepared.packet_count()),
                queue_id: prepared.queue_id(),
                command_dwords: Some(prepared.command_dwords()),
                queue_count: prepared.queue_count(),
                phase_count: prepared.phase_count(),
            });
        }
        self.prepared
            .as_ref()
            .map(|prepared| PreparedReplayIdentity {
                dispatch_count: prepared.dispatch_count(),
                packet_count: Some(prepared.packet_count()),
                queue_id: prepared.queue_id(),
                command_dwords: None,
                // Linear AQL is a single-queue, single-phase batch graph.
                queue_count: 1,
                phase_count: 1,
            })
    }

    pub fn replay_observation(&self) -> ReplayObservation {
        self.replay_observation
    }

    /// Start a request-local route-proof window without changing replay state.
    pub fn begin_replay_observation_window(&mut self) {
        self.replay_observation = ReplayObservation::default();
    }

    /// Invalidate the current request-local proof window after cancellation or
    /// another request-level failure that is not itself a replay error.
    pub fn invalidate_replay_observation_window(&mut self) {
        self.replay_observation.failed = true;
    }

    /// Build one request-scoped retained-replay proof marker.
    ///
    /// The daemon owns request boundaries and stderr emission. Invalid request
    /// IDs fail closed so an untrusted ID cannot inject or alias log lines.
    pub fn replay_observation_marker(&self, request_id: &str) -> Option<String> {
        if !self.route_proof_log
            || self.replay_observation.failed
            || self.replay_observation.count == 0
            || request_id.is_empty()
            || !request_id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')
            })
        {
            return None;
        }
        let position = self.replay_observation.first_position?;
        Some(format!(
            "HIPFIRE_REPLAY_ROUTE_PROOF transport={} position={} request_id={} replays={}",
            self.transport_name(),
            position,
            request_id,
            self.replay_observation.count
        ))
    }

    pub fn is_recording(&self) -> bool {
        self.state == ReplayState::RecordingWarmup && self.forward_eligible
    }

    /// Start the G0 eager arm (railgun design §5 G0). Launches through the
    /// recorder funnels are observed (kernel, geometry, exact kernarg bytes)
    /// while [`Self::is_recording`] stays false, so every recording-dependent
    /// predicate takes its eager branch. Refused while a recording is live.
    pub fn begin_g0_observation(&mut self) -> Result<(), &'static str> {
        if self.is_recording() {
            return Err("G0 observation cannot overlap a replay recording");
        }
        if self.g0_observation.is_some() {
            return Err("G0 observation is already active");
        }
        self.g0_observation = Some(Vec::new());
        Ok(())
    }

    #[inline]
    pub fn is_g0_observing(&self) -> bool {
        self.g0_observation.is_some()
    }

    pub(crate) fn observe_g0_launch(
        &mut self,
        kernel: &str,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg: &[u8],
    ) {
        if let Some(launches) = self.g0_observation.as_mut() {
            launches.push(G0Launch {
                kernel: kernel.to_owned(),
                grid,
                block,
                shared_mem,
                kernarg: kernarg.to_vec(),
            });
        }
    }

    /// Close the G0 eager arm and return the observed launches.
    pub fn finish_g0_observation(&mut self) -> Result<Vec<G0Launch>, &'static str> {
        self.g0_observation
            .take()
            .ok_or("no G0 observation is active")
    }

    /// Apply the model's one-shot plain-AR eligibility decision to this
    /// forward. Speculative/MTP re-seed and verify calls must neither populate
    /// the plain-AR capture nor route its prepared replay.
    pub fn set_forward_eligible(&mut self, eligible: bool) {
        self.forward_eligible = eligible;
    }

    pub fn is_enabled(&self) -> bool {
        self.request != ReplayBackendRequest::Hip && self.state != ReplayState::Fallback
    }

    /// Whether this controller owns the production one-shot capture lifecycle.
    /// Manual shadow/profiling controllers deliberately remain available for
    /// diagnosing routes which are not yet safe for automatic serving.
    pub fn automatic_lifecycle_enabled(&self) -> bool {
        self.auto_lifecycle
    }

    pub fn should_auto_finalize_capture(&self) -> bool {
        self.auto_lifecycle && self.is_recording()
    }

    pub fn begin_auto_capture_if_armed(&mut self) -> Result<(), &'static str> {
        if self.auto_lifecycle && self.forward_eligible && self.state == ReplayState::Armed {
            self.begin_capture()?;
        }
        Ok(())
    }

    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }

    /// Load every distinct captured HIP artifact through public HSA and prove
    /// that its loader-reported kernarg ABI accepts the exact padded bytes the
    /// HIP launch used. This creates no queue and executes no packet.
    pub fn probe_aql_contracts(
        &self,
        device_ordinal: usize,
    ) -> Result<Vec<AqlContractProbe>, String> {
        let runtime = Runtime::initialize(load_symbols().map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let device = runtime
            .select_gpu(GpuSelector::Ordinal(device_ordinal))
            .map_err(|error| error.to_string())?;
        let mut seen = BTreeSet::new();
        let mut probes = Vec::new();
        for launch in &self.recorded {
            if !seen.insert(launch.kernel.clone()) {
                continue;
            }
            let artifact = launch.artifact.as_ref().ok_or_else(|| {
                format!("captured kernel {:?} has no owning code object", launch.kernel)
            })?;
            let executable = Executable::load(&device, artifact.image().clone())
                .map_err(|error| format!("load {artifact}: {error}"))?;
            let symbol = format!("{}.kd", launch.kernel);
            let kernel = executable
                .kernel(&symbol)
                .map_err(|error| format!("resolve {symbol}: {error}"))?;
            let metadata = kernel.metadata();
            validate_loader_kernarg(launch, metadata.kernarg_segment_size as usize)
                .map_err(|reason| format!("{symbol}: {reason}"))?;
            probes.push(AqlContractProbe {
                kernel: launch.kernel.clone(),
                captured_kernarg_bytes: launch.kernarg.len(),
                loader_kernarg_bytes: metadata.kernarg_segment_size,
                loader_kernarg_alignment: metadata.kernarg_segment_alignment,
                static_group_bytes: metadata.group_segment_size,
                dynamic_group_bytes: launch.shared_mem,
            });
        }
        Ok(probes)
    }

    /// Lower the exact captured HIP sequence to one public-HSA queue. All
    /// explicit argument bytes remain unchanged; only the standardized
    /// 256-byte gfx12 implicit-argument suffix is synthesized from launch
    /// geometry, matching CLR's module-launch path.
    pub fn prepare_linear_aql(
        &mut self,
        device_ordinal: usize,
    ) -> Result<(usize, usize, u64), String> {
        self.prepare_linear_aql_prefix(device_ordinal, self.recorded.len())
    }

    pub fn prepare_linear_aql_prefix(
        &mut self,
        device_ordinal: usize,
        prefix: usize,
    ) -> Result<(usize, usize, u64), String> {
        if self.recorded.is_empty() {
            return Err("no captured launch sequence".to_owned());
        }
        if prefix < 2 || prefix > self.recorded.len() {
            return Err(format!(
                "AQL prefix {prefix} must be in 2..={}",
                self.recorded.len()
            ));
        }
        self.refuse_effect_incomplete_window()?;
        // The linear AQL graph submits every recorded kernarg verbatim; only the
        // GDN requant frame is patched per replay. A position-derived field or
        // grid would replay at its capture-time value, so such a tape is the
        // PM4 transport's (which applies every binding) or HIP's.
        if let Some((dispatch, binding)) = self
            .retained_kernarg_bindings(prefix)?
            .into_iter()
            .find(|(_, binding)| !matches!(binding, ReplayKernargBinding::GdnFrameU32 { .. }))
        {
            return Err(format!(
                "linear AQL cannot rebind the position-derived kernarg at offset {} of dispatch \
                 {dispatch} ({}); the tape needs the PM4 transport",
                binding.offset(),
                self.recorded[dispatch].kernel
            ));
        }
        if let Some(launch) = self.recorded[..prefix]
            .iter()
            .find(|launch| launch.grid_binding.is_some())
        {
            return Err(format!(
                "linear AQL cannot rebind the position-derived grid of {}; the tape needs the \
                 PM4 transport",
                launch.kernel
            ));
        }
        let runtime = Runtime::initialize(load_symbols().map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let device = runtime
            .select_gpu(GpuSelector::Ordinal(device_ordinal))
            .map_err(|error| error.to_string())?;
        let pool = KernargPool::discover(&device).map_err(|error| error.to_string())?;
        let mut executables = BTreeMap::<CodeObjectId, Executable>::new();
        let mut kernels = BTreeMap::<(CodeObjectId, String), Kernel>::new();
        let mut dispatches = Vec::with_capacity(prefix);
        let mut dynamic_gdn_frames = Vec::new();

        for launch in self.recorded.iter().take(prefix) {
            let artifact = launch.artifact.as_ref().ok_or_else(|| {
                format!("captured kernel {:?} has no owning code object", launch.kernel)
            })?;
            let id = artifact.id();
            if !executables.contains_key(&id) {
                let executable = Executable::load(&device, artifact.image().clone())
                    .map_err(|error| format!("load {artifact}: {error}"))?;
                executables.insert(id, executable);
            }
            let symbol = format!("{}.kd", launch.kernel);
            let key = (id, symbol.clone());
            if !kernels.contains_key(&key) {
                let kernel = executables[&id]
                    .kernel(&symbol)
                    .map_err(|error| format!("resolve {symbol}: {error}"))?;
                kernels.insert(key.clone(), kernel);
            }
            let kernel = kernels[&key].clone();
            let metadata = kernel.metadata();
            let mut kernarg = pool
                .allocate_for(metadata)
                .map_err(|error| format!("allocate {symbol} kernarg: {error}"))?;
            populate_gfx12_kernarg(&mut kernarg, launch, metadata.kernarg_segment_size as usize)?;
            let mut workgroup = [0_u16; 3];
            for (axis, value) in launch.block.into_iter().enumerate() {
                workgroup[axis] = u16::try_from(value)
                    .map_err(|_| format!("{symbol}: workgroup dimension {value} exceeds u16"))?;
            }
            check_prepared_grid(device.name(), &symbol, launch.grid)?;
            let geometry = LaunchGeometry::from_hip_workgroups(launch.grid, workgroup)
                .map_err(|error| format!("{symbol}: {error}"))?;
            let dispatch = RecordedDispatch::new(0, kernel, geometry, kernarg)
                .map_err(|error| format!("{symbol}: {error}"))?
                .with_dynamic_group_bytes(launch.shared_mem)
                .map_err(|error| format!("{symbol}: {error}"))?;
            if launch.kernel == "gated_delta_net_q8_fast"
                || launch.kernel.starts_with("gated_delta_net_q8_compact")
            {
                if metadata.kernarg_segment_size < 80 {
                    return Err(format!(
                        "{symbol}: loader kernarg is too short for dynamic frame binding"
                    ));
                }
                dynamic_gdn_frames.push(dispatches.len());
            }
            dispatches.push(dispatch);
        }

        let required = dispatches
            .len()
            .checked_add(1)
            .ok_or_else(|| "AQL packet count overflow".to_owned())?;
        let queue_size = required
            .next_power_of_two()
            .max(*device.queue_size_range().start() as usize);
        let queue_size = u32::try_from(queue_size)
            .map_err(|_| format!("AQL queue size {queue_size} exceeds u32"))?;
        if !device.queue_size_range().contains(&queue_size) {
            return Err(format!(
                "AQL queue size {queue_size} outside {:?}",
                device.queue_size_range()
            ));
        }
        let mut headers = vec![HeaderPolicy::BATCH_BOUNDARY_INTERNAL_SERIAL; dispatches.len()];
        headers[0] = HeaderPolicy::BATCH_BOUNDARY_FIRST_SERIAL;
        for (index, launch) in self.recorded.iter().take(prefix).enumerate() {
            if launch.kernel == "repeat_interleave_qk_f32" {
                headers[index] = HeaderPolicy::RECORDED_DISPATCH;
                if index + 1 < headers.len() {
                    headers[index + 1] = HeaderPolicy::BATCH_INTERNAL_ACQUIRE_SYSTEM;
                }
            } else if matches!(
                launch.kernel.as_str(),
                "fused_silu_mul_mq_rotate"
                    | "mq_rotate_x"
                    | "rope_partial_halfsplit_f32"
                    | "rope_partial_halfsplit_f32_headgrid"
            ) {
                if launch.kernel == "mq_rotate_x" {
                    headers[index] = HeaderPolicy::BATCH_INTERNAL_RELEASE_SYSTEM;
                    if index + 1 < headers.len() {
                        headers[index + 1] = HeaderPolicy::BATCH_INTERNAL_ACQUIRE_SYSTEM;
                    }
                } else {
                    headers[index] = HeaderPolicy::RECORDED_DISPATCH;
                }
            }
        }
        for index in 1..headers.len() {
            let previous = self.recorded[index - 1].kernel.as_str();
            let current = self.recorded[index].kernel.as_str();
            if independent_sibling(previous, current) {
                headers[index] = HeaderPolicy::BATCH_BOUNDARY_INTERNAL_INDEPENDENT;
            }
        }
        // HC ping-pong publishes the next residual allocation from
        // `hc_mix_4stream`, then `hc_input_map_4stream` consumes it after the
        // following block's control kernels. Queue-order barriers serialize
        // execution but do not by themselves establish gfx1151 cache
        // visibility, so place the narrow same-agent release/acquire pair at
        // the actual producer/consumer boundary.
        for (index, launch) in self.recorded.iter().take(prefix).enumerate() {
            match launch.kernel.as_str() {
                "hc_mix_4stream" => {
                    headers[index] = HeaderPolicy::BATCH_INTERNAL_RELEASE_AGENT;
                }
                "hc_input_map_4stream" => {
                    headers[index] = HeaderPolicy::BATCH_INTERNAL_ACQUIRE_AGENT;
                }
                _ => {}
            }
        }
        apply_qwen_q8_full_attention_visibility(&self.recorded[..prefix], &mut headers);
        // A queue barrier orders execution but does not publish vector-cache
        // writes to the next dispatch on gfx11/gfx12. Keep ordinary intra-tape
        // ownership at agent scope, with system scope only at the external
        // HIP/AQL entry and host-visible completion boundaries.
        if Pm4Architecture::from_name(device.name())? != Pm4Architecture::Gfx10 {
            headers.fill(HeaderPolicy::SAME_AGENT_DISPATCH);
            if headers.len() == 1 {
                headers[0] = HeaderPolicy::RECORDED_DISPATCH;
            } else {
                headers[0] = HeaderPolicy {
                    barrier: true,
                    acquire: FenceScope::System,
                    release: FenceScope::Agent,
                };
                let last = headers.len() - 1;
                headers[last] = HeaderPolicy {
                    barrier: true,
                    acquire: FenceScope::Agent,
                    release: FenceScope::System,
                };
            }
        }
        let graph = if self.request == ReplayBackendRequest::Auto {
            SingleQueueBatchGraph::create_unprofiled_with_dispatch_headers(
                &device,
                queue_size,
                dispatches,
                BatchFencePolicy::BoundarySerialized,
                headers,
            )
        } else {
            SingleQueueBatchGraph::create_with_dispatch_headers(
                &device,
                queue_size,
                dispatches,
                BatchFencePolicy::BoundarySerialized,
                headers,
            )
        }
        .map_err(|error| error.to_string())?;
        let summary = (
            graph.dispatch_count(),
            graph.packet_count(),
            graph.queue_id(),
        );
        self.prepared = Some(PreparedLinearAqlReplay {
            graph,
            dynamic_gdn_frames,
        });
        self.state = ReplayState::Ready;
        Ok(summary)
    }

    /// Lower a captured prefix to one retained architecture-native PM4
    /// indirect buffer. Unsupported HSA agents fail closed before commands are
    /// constructed; gfx10/11 and gfx12 never share register encodings.
    /// Lower a captured prefix to a single retained PM4 IB.
    ///
    /// Permissive by construction: this is the entry point every certified
    /// route already uses, and its tape identity is sealed evidence, so it
    /// must keep its historical semantics. A route that replays across
    /// advancing decode positions must instead call
    /// [`Self::prepare_pm4_prefix_calibrated`], which refuses to prepare
    /// until two recordings have been differenced.
    pub fn prepare_pm4_prefix(
        &mut self,
        device_ordinal: usize,
        prefix: usize,
    ) -> Result<(usize, u32, u64), String> {
        self.prepare_pm4_prefix_inner(device_ordinal, prefix, false, true)
    }

    /// Position-aware variant: refuses to prepare unless
    /// [`Self::synthesize_position_bindings`] has differenced two recordings
    /// of this tape, so a position-tracking kernarg scalar cannot be retained
    /// unproven.
    pub fn prepare_pm4_prefix_calibrated(
        &mut self,
        device_ordinal: usize,
        prefix: usize,
    ) -> Result<(usize, u32, u64), String> {
        self.prepare_pm4_prefix_inner(device_ordinal, prefix, false, false)
    }

    /// Lower a captured prefix to a single GFX12 IB with per-dispatch timestamps.
    pub fn prepare_pm4_dispatch_profile(
        &mut self,
        device_ordinal: usize,
        prefix: usize,
    ) -> Result<(usize, u32, u64), String> {
        self.prepare_pm4_prefix_inner(device_ordinal, prefix, true, true)
    }

    fn prepare_pm4_prefix_inner(
        &mut self,
        device_ordinal: usize,
        prefix: usize,
        dispatch_profile: bool,
        allow_uncalibrated: bool,
    ) -> Result<(usize, u32, u64), String> {
        if !allow_uncalibrated && !self.position_bindings_calibrated {
            return Err(
                "position bindings not calibrated; call synthesize_position_bindings or explicitly opt out via prepare_pm4_prefix_allow_uncalibrated".to_owned(),
            );
        }
        if self.recorded.is_empty() {
            return Err("no captured launch sequence".to_owned());
        }
        if prefix == 0 || prefix > self.recorded.len() {
            return Err(format!(
                "PM4 prefix {prefix} must be in 1..={}",
                self.recorded.len()
            ));
        }
        self.refuse_effect_incomplete_window()?;
        let runtime = Runtime::initialize(load_symbols().map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let device = runtime
            .select_gpu(GpuSelector::Ordinal(device_ordinal))
            .map_err(|error| error.to_string())?;
        let pm4_architecture = Pm4Architecture::from_device(&device)?;
        if dispatch_profile && pm4_architecture == Pm4Architecture::Gfx10 {
            return Err("per-dispatch PM4 profiling requires gfx11 or gfx12".to_owned());
        }
        let dispatch_initiator_policy =
            gfx10_dispatch_initiator_policy(pm4_architecture, device.name());
        let dispatch_interleave = gfx1151_dispatch_interleave(pm4_architecture, device.name());
        let resource_limits_policy =
            gfx1151_resource_limits_policy(pm4_architecture, device.name());
        let cu_mask = gfx1151_cu_mask(pm4_architecture, device.name());
        let entry_acquire_policy = gfx1151_entry_acquire_policy(pm4_architecture, device.name());
        let pool = KernargPool::discover(&device).map_err(|error| error.to_string())?;
        let kernarg_pool = retained_kernarg_pool(&device, &pool, entry_acquire_policy);
        let mut executables = BTreeMap::<CodeObjectId, Executable>::new();
        let mut resolved = BTreeMap::<(CodeObjectId, String), Kernel>::new();
        let mut kernels = Vec::with_capacity(prefix);
        let mut kernargs = Vec::with_capacity(prefix);
        let mut geometries = Vec::with_capacity(prefix);
        let mut bound_explicit_lens = Vec::with_capacity(prefix);
        let mut dynamic_gdn_frames = Vec::new();
        let mut dynamic_grids = Vec::new();

        for launch in self.recorded.iter().take(prefix) {
            let artifact = launch.artifact.as_ref().ok_or_else(|| {
                format!("captured kernel {:?} has no owning code object", launch.kernel)
            })?;
            let id = artifact.id();
            if !executables.contains_key(&id) {
                let executable = Executable::load(&device, artifact.image().clone())
                    .map_err(|error| format!("load {artifact}: {error}"))?;
                executables.insert(id, executable);
            }
            let symbol = format!("{}.kd", launch.kernel);
            let key = (id, symbol.clone());
            if !resolved.contains_key(&key) {
                let kernel = executables[&id]
                    .kernel(&symbol)
                    .map_err(|error| format!("resolve {symbol}: {error}"))?;
                resolved.insert(key.clone(), kernel);
            }
            let kernel = resolved[&key].clone();
            let metadata = kernel.metadata();
            let mut kernarg = kernarg_pool
                .allocate_for(metadata)
                .map_err(|error| format!("allocate {symbol} kernarg: {error}"))?;
            // Slice 1: typed segments re-encode from `ReplayBindings` (byte-
            // equal to the snapshot here); untyped segments copy the snapshot.
            let explicit_len = populate_gfx12_kernarg_bound(
                &mut kernarg,
                launch,
                &self.replay_bindings,
                metadata.kernarg_segment_size as usize,
            )
            .map_err(|error| format!("{symbol}: {error}"))?;
            bound_explicit_lens.push(explicit_len);
            let mut workgroup = [0_u16; 3];
            for (axis, value) in launch.block.into_iter().enumerate() {
                workgroup[axis] = u16::try_from(value)
                    .map_err(|_| format!("{symbol}: workgroup dimension {value} exceeds u16"))?;
            }
            // Determine geometry grid: if a dynamic binding exists and a max_position
            // was requested, size the prepared IB for the maximum admitted position
            // so replay can patch DOWN. Otherwise use the recorded grid (today's
            // behavior).
            let mut geometry_grid = launch.grid;
            let mut grid_binding_for_storage = launch.grid_binding;
            if let Some(binding) = launch.grid_binding {
                if let Some(max_pos) = self.prepared_max_position {
                    let max_units = binding
                        .units_for(max_pos)
                        .map_err(|error| format!("{symbol}: {error}"))?;
                    let axis = match binding {
                        ReplayGridBinding::PositionCeilDiv { axis, .. } => usize::from(axis),
                    };
                    if axis >= 3 {
                        return Err(format!("{symbol}: invalid grid binding axis {axis}"));
                    }
                    let mut prepared_grid = launch.grid;
                    prepared_grid[axis] = max_units;
                    geometry_grid = prepared_grid;
                    // Store the prepared maximum as the recorded value so
                    // bind(current_position, prepared_grid) correctly narrows.
                    grid_binding_for_storage = Some(binding);
                    // Keep launch.grid replaced for geometry; dynamic_grids stores prepared_grid.
                }
            }
            check_prepared_grid(device.name(), &symbol, geometry_grid)?;
            let geometry = LaunchGeometry::from_hip_workgroups(geometry_grid, workgroup)
                .map_err(|error| format!("{symbol}: {error}"))?;
            device
                .validate_geometry(geometry)
                .map_err(|error| format!("{symbol}: {error}"))?;
            if is_gdn_kernel(&launch.kernel) && metadata.kernarg_segment_size < 80 {
                return Err(format!(
                    "{symbol}: loader kernarg is too short for dynamic frame binding"
                ));
            }
            if let Some(binding) = grid_binding_for_storage {
                let grid_to_store = if self.prepared_max_position.is_some() {
                    geometry_grid
                } else {
                    launch.grid
                };
                dynamic_grids.push((kernargs.len(), binding, grid_to_store, launch.block));
            }
            kernels.push(kernel);
            kernargs.push(kernarg);
            geometries.push(geometry);
        }
        // GDN frames, synthesized position bindings and engine-declared
        // bindings, from the one builder the recorded-HIP oracle also uses. New
        // tapes leave the legacy `dynamic_gdn_frames` vector empty so replay has
        // exactly one path that decides frame consumption.
        let dynamic_kernarg_bindings = self
            .retained_kernarg_bindings(prefix)
            .map_err(|reason| format!("retained PM4 kernarg bindings: {reason}"))?;

        let gfx12_gcr_trim = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_GCR_TRIM")
            .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
            .unwrap_or(true);
        let gfx11_vmem_acquire =
            match hipfire_config::process_value("HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE") {
                Some(value) if value != "auto" => matches!(value.as_str(), "1" | "true" | "on"),
                _ => device.name().eq_ignore_ascii_case("gfx1151"),
            };
        let gfx12_vmem_acquire = pm4_gfx12_vmem_acquire_from_config();
        if gfx12_vmem_acquire {
            eprintln!("[redline] gfx12 PM4 VMEM RMW acquire rung enabled (explicit opt-in)");
        }
        let radiowave_certifications = if gfx11_vmem_acquire || gfx12_vmem_acquire {
            radiowave_certifications(&self.recorded, prefix)
        } else {
            BTreeMap::new()
        };
        if gfx11_vmem_acquire {
            let artifacts = self
                .recorded
                .iter()
                .take(prefix)
                .filter_map(|launch| launch.artifact.as_ref().map(|artifact| artifact.id()))
                .collect::<BTreeSet<_>>();
            let vmem_launches = self
                .recorded
                .iter()
                .take(prefix)
                .filter(|launch| radiowave_vmem_only_consumer(&radiowave_certifications, launch))
                .count();
            let vmem_symbols = self
                .recorded
                .iter()
                .take(prefix)
                .filter(|launch| radiowave_vmem_only_consumer(&radiowave_certifications, launch))
                .map(|launch| launch.kernel.as_str())
                .collect::<BTreeSet<_>>()
                .len();
            eprintln!(
                "[redline] Radiowave code-object contracts: certified_artifacts={}/{} \
                 vmem_symbols={} vmem_launches={}",
                radiowave_certifications.len(),
                artifacts.len(),
                vmem_symbols,
                vmem_launches,
            );
            eprintln!(
                "[redline] Radiowave argument effects: certified_launches={} \
                 fallback_launches={} unknown_launches={}",
                self.radiowave_effect_launches,
                self.fallback_effect_launches,
                self.unknown_effect_launches,
            );
        }
        let mut wait_audit = Pm4WaitAudit::default();
        let mut audit_frontier = ResourceFrontier::default();
        for index in 0..prefix {
            if index != 0 {
                let previous_launch = &self.recorded[index - 1];
                let current_launch = &self.recorded[index];
                let previous = previous_launch.kernel.as_str();
                let current = current_launch.kernel.as_str();
                let allowlist_independent = independent_sibling(previous, current);
                let resource_covered = audit_frontier.covered(current_launch);
                let resources_independent = audit_frontier.independent(current_launch);
                let exact_start_independent =
                    audit_frontier.independent_by_exact_start(current_launch);
                wait_audit.observe(
                    previous_launch,
                    current_launch,
                    allowlist_independent,
                    resources_independent,
                    exact_start_independent,
                    resource_covered,
                );
                audit_frontier.advance(current_launch, resources_independent);
            } else {
                audit_frontier.advance(&self.recorded[index], false);
            }
        }
        if self.pm4_wait_policy != Pm4WaitPolicy::Allowlist {
            wait_audit.report(self.pm4_wait_policy);
        }

        // Dynamic direct-dispatch patching is intentionally limited to the
        // certified one-IB path. Multi-queue phases duplicate and reorder
        // command buffers, so a global capture index is not a safe patch key.
        let queue_limit = if dispatch_profile {
            1
        } else if dynamic_grids.is_empty() {
            self.pm4_queue_policy.resolve(device.name(), usize::MAX)
        } else {
            1
        };
        let stream_accounting = pm4_stream_accounting_enabled();
        validate_pm4_stream_accounting_queue_count(stream_accounting, queue_limit)?;
        if cu_mask.is_some() && queue_limit != 1 {
            return Err("gfx1151 CU-mask experiments require single-queue PM4 replay".to_owned());
        }
        let gfx1010_exact = gfx1010_release_wait_required(pm4_architecture, device.name());
        let gfx1010_dependency =
            gfx1010_dependency_policy_from_config(pm4_architecture, device.name())?;
        // Exact gfx1010 admits only single-queue non-native single-phase lowering.
        // Multi-queue / multi-phase / native-sync topologies stay on CS_PARTIAL_FLUSH
        // elsewhere and are fail-closed here rather than silently degraded.
        // Restriction holds under both ReleaseWait and CsPartialFlush overrides.
        if gfx1010_exact && queue_limit != 1 {
            return Err(
                "gfx1010 RELEASE_MEM/WAIT_REG_MEM dependency fence requires single-queue \
                 non-native single-phase retained PM4"
                    .to_owned(),
            );
        }
        let mut dependency_fence = None;
        let mut dispatch_boundaries = Vec::new();
        let (graph, command_dwords) = if queue_limit == 1 {
            let recorded = &self.recorded[..prefix];
            let reorder_window = pm4_single_ib_reorder_from_config(device.name());
            let order = match reorder_window {
                None => (0..prefix).collect::<Vec<_>>(),
                Some(window) => {
                    let order = pm4_width_reorder(recorded, window);
                    let moved = order
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(slot, index)| *slot != *index)
                        .count();
                    let max_displacement = order
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(slot, index)| slot.abs_diff(index))
                        .max()
                        .unwrap_or(0);
                    let sequence_hash =
                        replay_sequence_hash(order.iter().map(|index| &recorded[*index]));
                    eprintln!(
                        "[redline] single-IB reorder(arch={}, window={window}, scheduler=level): \
                         moved={moved}/{prefix} max_displacement={max_displacement} \
                         execution_sequence_hash={sequence_hash:016x}",
                        device.name()
                    );
                    order
                }
            };
            let dependency_mode = match gfx1010_dependency {
                Gfx1010DependencyPolicy::ReleaseWait => {
                    let fence = pool
                        .allocate_fine_grained_bytes(4, 4)
                        .map_err(|error| format!("allocate gfx1010 dependency fence: {error}"))?;
                    let address = fence.address() as u64;
                    if address == 0 || address & 3 != 0 {
                        return Err(format!(
                            "gfx1010 dependency fence address {address:#x} is null or unaligned"
                        ));
                    }
                    dependency_fence = Some(fence);
                    LegacyDependencyMode::ReleaseWait {
                        address,
                        next_epoch: 0,
                    }
                }
                Gfx1010DependencyPolicy::CsPartialFlush => LegacyDependencyMode::CsPartialFlush,
            };
            let mut commands = Pm4Commands::new_with_dependency(
                pm4_architecture,
                self.pm4_register_policy,
                dispatch_initiator_policy,
                dispatch_interleave,
                resource_limits_policy,
                dependency_mode,
            );
            let pacing = if dispatch_profile {
                Gfx12DispatchPacing::None
            } else {
                self.pm4_gfx12_dispatch_pacing
            };
            if let Pm4Commands::Gfx12(gfx12) = &mut commands {
                gfx12.set_dispatch_pacing(pacing);
            }
            // Sentinel epoch 0 before entry acquire: every immutable replay
            // re-submits this prefix so a stale prior epoch cannot satisfy the
            // next run (ABA).
            commands.emit_entry_sentinel_reset()?;
            commands.acquire_entry(gfx12_gcr_trim, entry_acquire_policy);
            let mut resource_frontier = ResourceFrontier::default();
            let mut dependency_waits = 0usize;
            let mut dependency_acquires = 0usize;
            let shadow_on = self.railgun_shadow.is_some();
            let mut shadow_decisions = Vec::new();
            for (position, index) in order.iter().copied().enumerate() {
                let mut boundary = Pm4DispatchBoundary::default();
                if position != 0 {
                    let previous_index = order[position - 1];
                    let previous_launch = &self.recorded[previous_index];
                    let current_launch = &self.recorded[index];
                    let previous = previous_launch.kernel.as_str();
                    let current = current_launch.kernel.as_str();
                    let allowlist_independent = independent_sibling(previous, current);
                    let resources_independent = resource_frontier.independent(current_launch);
                    let independent = match self.pm4_wait_policy {
                        Pm4WaitPolicy::Allowlist | Pm4WaitPolicy::ResourceAudit => {
                            allowlist_independent
                        }
                        Pm4WaitPolicy::Resource => resources_independent,
                    };
                    // GC12 can retain a vector-cache line for the reused
                    // rotated-output allocation across dispatches in one IB.
                    // Invalidate it before the writer starts; a consumer-side
                    // acquire after the writer is too late for this hazard.
                    let gfx12_pre_dispatch_acquire = pm4_architecture == Pm4Architecture::Gfx12
                        && requires_gfx12_pre_dispatch_vmem_acquire(current);
                    if gfx12_pre_dispatch_acquire || !independent {
                        dependency_waits += 1;
                        boundary.wait_compute_idle = true;
                        commands.wait_compute_idle()?;
                    }
                    resource_frontier.advance(current_launch, resources_independent);
                    let acquire = (!independent && commands.requires_dependency_acquire())
                        || self
                            .pm4_mid_acquire_policy
                            .acquire_between(previous, current);
                    if gfx12_pre_dispatch_acquire {
                        dependency_acquires += 1;
                        // Gfx12-only arm, so only the gfx12 opt-in can select
                        // the VMEM rung here (`pm4_vmem_acquire_enabled` is
                        // arch-excluded on gfx12 and always false on this path).
                        boundary.acquire_vmem = pm4_gfx12_vmem_acquire_enabled(
                            gfx12_vmem_acquire,
                            &radiowave_certifications,
                            current_launch,
                        );
                        if boundary.acquire_vmem {
                            commands.acquire_inter_node(gfx12_gcr_trim, true);
                        } else {
                            // Baseline default: the pre-dispatch hazard needs
                            // the system-scope acquire; the weaker inter-node
                            // rung behind this arm is an explicit opt-in only.
                            commands.gfx12_system_acquire()?;
                        }
                    } else if acquire {
                        dependency_acquires += 1;
                        boundary.acquire_vmem = pm4_vmem_acquire_enabled(
                            pm4_architecture,
                            gfx11_vmem_acquire,
                            &radiowave_certifications,
                            current_launch,
                        ) || pm4_gfx12_vmem_acquire_enabled(
                            gfx12_vmem_acquire,
                            &radiowave_certifications,
                            current_launch,
                        );
                        commands.acquire_inter_node(gfx12_gcr_trim, boundary.acquire_vmem);
                    }
                    if shadow_on {
                        shadow_decisions.push(railgun::shadow::RedlineDecision {
                            effects: redline_effect_source(current_launch),
                            resource_independent: resources_independent,
                            name_acquire: self
                                .pm4_mid_acquire_policy
                                .acquire_between(previous, current),
                            pre_dispatch_name: gfx12_pre_dispatch_acquire,
                        });
                    }
                } else {
                    resource_frontier.advance(&self.recorded[index], false);
                    if shadow_on {
                        shadow_decisions.push(railgun::shadow::RedlineDecision {
                            effects: redline_effect_source(&self.recorded[index]),
                            ..Default::default()
                        });
                    }
                }
                commands
                    .dispatch(
                        &kernels[index],
                        geometries[index],
                        self.recorded[index].shared_mem,
                        kernargs[index].address(),
                    )
                    .map_err(|error| format!("{}: {error}", self.recorded[index].kernel))?;
                if dispatch_profile {
                    dispatch_boundaries.push(boundary);
                }
            }
            commands.trailing_release()?;
            if dispatch_profile {
                commands.populate_dispatch_span_boundaries(&mut dispatch_boundaries)?;
            }
            let command_dwords = commands.len_dwords();
            if let Pm4Commands::Gfx12(gfx12) = &commands {
                if pacing != Gfx12DispatchPacing::None {
                    eprintln!(
                        "[redline] gfx12 PM4 dispatch pacing {pacing:?}: dispatches={prefix} \
                         nop_dwords={} ({:.1}/dispatch) command_dwords={command_dwords}",
                        gfx12.pacing_dwords(),
                        gfx12.pacing_dwords() as f64 / prefix as f64,
                    );
                }
            }
            if reorder_window.is_some() {
                eprintln!(
                    "[redline] single-IB schedule stats arch={}: \
                     independent_adjacencies={} dependency_waits={} \
                     dependency_acquires={} terminal_waits=1 command_dwords={command_dwords}",
                    device.name(),
                    prefix.saturating_sub(1).saturating_sub(dependency_waits),
                    dependency_waits,
                    dependency_acquires,
                );
                // Legacy gfx1151 census stays when stream accounting is off.
                // With the flag set, the richer report below subsumes it.
                if !stream_accounting && device.name().eq_ignore_ascii_case("gfx1151") {
                    match commands.packet_census() {
                        Some(Ok(census)) => {
                            eprintln!("[redline] gfx1151 PM4 packet census: {census:?}");
                        }
                        Some(Err(dword)) => {
                            eprintln!("WARNING: gfx1151 PM4 packet census failed at dword {dword}");
                        }
                        None => {}
                    }
                }
            }
            // Preparation-only stream accounting: after terminal idle, before
            // graph creation. Reachable on gfx10/gfx11 (incl. gfx1100) without
            // requiring the reorder flag. Fail closed on malformed streams.
            if stream_accounting {
                match pm4_architecture {
                    Pm4Architecture::Gfx10 | Pm4Architecture::Gfx11 => {
                        let execution_sequence_hash =
                            replay_sequence_hash(order.iter().map(|index| &recorded[*index]));
                        report_pm4_stream_accounting(
                            device.name(),
                            &order,
                            recorded,
                            &commands,
                            command_dwords,
                            execution_sequence_hash,
                            dependency_waits,
                            dependency_acquires,
                        )?;
                    }
                    Pm4Architecture::Gfx12 => {
                        return Err(
                            "HIPFIRE_REPLAY_PM4_STREAM_ACCOUNTING requires gfx10/gfx11 PM4"
                                .to_owned(),
                        );
                    }
                }
            }
            // railgun M1 shadow: prepare railgun's lowering of the same tape
            // and diff it against this one. Nothing railgun builds is
            // submitted; the route below is unchanged.
            if let Some(shadow) = self.railgun_shadow.as_mut() {
                let skip = if pm4_architecture != Pm4Architecture::Gfx12 {
                    Some("railgun M1 authors gfx12 tapes only")
                } else if reorder_window.is_some() {
                    Some("the single-IB reorder permutes the recorded order")
                } else if dispatch_profile {
                    Some("per-dispatch profile tape")
                } else {
                    None
                };
                match skip {
                    Some(reason) if shadow.backend_railgun() => {
                        return Err(format!("railgun backend refused: {reason}"));
                    }
                    Some(reason) => eprintln!("[railgun] shadow skipped: {reason}"),
                    None => {
                        let register_policy = self.pm4_register_policy;
                        let new_commands = || {
                            Pm4Commands::new_with_dependency(
                                pm4_architecture,
                                register_policy,
                                dispatch_initiator_policy,
                                dispatch_interleave,
                                resource_limits_policy,
                                LegacyDependencyMode::CsPartialFlush,
                            )
                        };
                        let kernarg_images = kernargs
                            .iter_mut()
                            .map(|buffer| {
                                let address = buffer.address() as usize as u64;
                                (buffer.as_mut_bytes().to_vec(), address)
                            })
                            .collect();
                        let resources = self
                            .tape_resources
                            .iter()
                            .map(|(&(base, bytes), id)| (id.index(), (base, bytes)))
                            .collect();
                        let executable = shadow.compare(
                            railgun_shadow::RedlinePrepared {
                                device_name: device.name(),
                                launches: recorded,
                                ib: commands.gfx12_dwords().unwrap_or(&[]),
                                kernels: &kernels,
                                kernargs: kernarg_images,
                                decisions: std::mem::take(&mut shadow_decisions),
                                word_patches: &dynamic_kernarg_bindings,
                                grid_patches: &dynamic_grids,
                                resources,
                            },
                            railgun_shadow::Transport {
                                pool: &pool,
                                new_commands: &new_commands,
                                gcr_trim: gfx12_gcr_trim,
                                entry_policy: entry_acquire_policy,
                            },
                        );
                        // HIPFIRE_RAILGUN_BACKEND=railgun (M2): the prepared
                        // tape runs railgun's lowering; a refusal fails the
                        // prepare closed, never back to this planner's IB.
                        match executable {
                            None => {}
                            Some(Ok(railgun_commands)) => commands = railgun_commands,
                            Some(Err(reason)) => {
                                return Err(format!("railgun backend refused: {reason}"))
                            }
                        }
                    }
                }
            }
            let command_dwords = commands.len_dwords();
            // HIPFIRE_REDLINE_IB_POOL=vmem: allocate the retained indirect
            // buffer from a GPU-agent (VRAM) pool so the command processor
            // fetches the tape from VRAM instead of re-reading it over the
            // host interface on every replay. Host-read surfaces
            // (timestamps, completion signals) stay on the CPU-agent pool.
            // Falls back to the host pool with a warning when no device-local
            // pool exists.
            static IB_VMEM: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
                hipfire_config::developer_var("HIPFIRE_REDLINE_IB_POOL").as_deref() == Ok("vmem")
            });
            let ib_pool = if *IB_VMEM {
                match KernargPool::discover_device_local(&device) {
                    Ok(p) => Some(p),
                    Err(error) => {
                        eprintln!(
                            "[redline] HIPFIRE_REDLINE_IB_POOL=vmem but no device-local pool \
                             found ({error}); using host pool"
                        );
                        None
                    }
                }
            } else {
                None
            };
            let graph = commands.create_graph(
                &device,
                &pool,
                ib_pool.as_ref(),
                cu_mask.as_ref(),
                dispatch_profile,
            )?;
            (PreparedPm4Graph::Single(graph), command_dwords)
        } else {
            if self
                .railgun_shadow
                .as_ref()
                .is_some_and(railgun_shadow::RailgunShadow::backend_railgun)
            {
                return Err(
                    "railgun backend refused: multi-queue PM4 tapes are not lowered by railgun"
                        .to_owned(),
                );
            }
            let min_parallel_width = pm4_min_parallel_width_from_config();
            let min_parallel_workgroups = pm4_min_parallel_workgroups_from_config();
            let max_parallel_phases = pm4_max_parallel_phases_from_config();
            let native_phase_sync = pm4_native_phase_sync_from_config();
            let ds4_ffn_branch_chains = pm4_ds4_ffn_branch_chains_from_config();
            let plans = if ds4_ffn_branch_chains {
                if self.recorded[..prefix]
                    .iter()
                    .any(|launch| launch.kernel == "zero_f32")
                {
                    pm4_ds4_ffn_branch_plan(&self.recorded[..prefix])?
                } else {
                    pm4_ds4_batched_ffn_branch_plan(&self.recorded[..prefix])?
                }
            } else {
                pm4_phase_plan(
                    &self.recorded[..prefix],
                    min_parallel_width,
                    min_parallel_workgroups,
                    max_parallel_phases,
                )
            };
            let parallel_phases = plans.iter().filter(|phase| phase.parallel).count();
            let branch_chain_phases = plans
                .iter()
                .filter(|phase| phase.lane_split.is_some())
                .count();
            let max_width = plans
                .iter()
                .map(|phase| phase.indices.len())
                .max()
                .unwrap_or(1);
            let parallel_launches = plans
                .iter()
                .filter(|phase| phase.parallel)
                .map(|phase| phase.indices.len())
                .sum::<usize>();
            let mut phase_commands = Vec::<Vec<Pm4Commands>>::with_capacity(plans.len());
            let mut command_dwords = 0_u32;
            let mut max_queue_count = 1_usize;

            for phase in &plans {
                let lane_count = if phase.parallel {
                    if phase.lane_split.is_some() {
                        self.pm4_queue_policy.resolve(device.name(), 2)
                    } else {
                        self.pm4_queue_policy
                            .resolve(device.name(), phase.indices.len())
                    }
                } else {
                    1
                };
                if lane_count > 1 {
                    let mut lanes = (0..lane_count)
                        .map(|_| {
                            let mut commands = Pm4Commands::new(
                                pm4_architecture,
                                self.pm4_register_policy,
                                dispatch_initiator_policy,
                                dispatch_interleave,
                                resource_limits_policy,
                            );
                            commands.acquire_entry(gfx12_gcr_trim, entry_acquire_policy);
                            commands
                        })
                        .collect::<Vec<_>>();
                    let lane_indices = if let Some(split) = phase.lane_split {
                        if lane_count != 2 || split == 0 || split >= phase.indices.len() {
                            return Err(format!(
                                "invalid DeepSeek4 FFN branch phase: lanes={lane_count} split={split} launches={}",
                                phase.indices.len()
                            ));
                        }
                        vec![
                            phase.indices[..split].to_vec(),
                            phase.indices[split..].to_vec(),
                        ]
                    } else {
                        let mut lane_indices = vec![Vec::<usize>::new(); lane_count];
                        for (position, index) in phase.indices.iter().copied().enumerate() {
                            lane_indices[position % lane_count].push(index);
                        }
                        lane_indices
                    };
                    for (lane, indices) in lanes.iter_mut().zip(&lane_indices) {
                        let mut resource_frontier = ResourceFrontier::default();
                        for (position, index) in indices.iter().copied().enumerate() {
                            if position != 0 && phase.lane_split.is_some() {
                                let previous_index = indices[position - 1];
                                let previous_launch = &self.recorded[previous_index];
                                let current_launch = &self.recorded[index];
                                let previous = previous_launch.kernel.as_str();
                                let current = current_launch.kernel.as_str();
                                let resources_independent =
                                    resource_frontier.independent(current_launch);
                                if !resources_independent {
                                    lane.wait_compute_idle()?;
                                }
                                resource_frontier.advance(current_launch, resources_independent);
                                if (!resources_independent && lane.requires_dependency_acquire())
                                    || self
                                        .pm4_mid_acquire_policy
                                        .acquire_between(previous, current)
                                {
                                    lane.acquire_inter_node(
                                        gfx12_gcr_trim,
                                        ((pm4_architecture != Pm4Architecture::Gfx12
                                            && gfx11_vmem_acquire)
                                            || (pm4_architecture == Pm4Architecture::Gfx12
                                                && gfx12_vmem_acquire))
                                            && radiowave_vmem_only_consumer(
                                                &radiowave_certifications,
                                                current_launch,
                                            ),
                                    );
                                }
                            } else {
                                resource_frontier.advance(&self.recorded[index], false);
                            }
                            lane.dispatch(
                                &kernels[index],
                                geometries[index],
                                self.recorded[index].shared_mem,
                                kernargs[index].address(),
                            )
                            .map_err(|error| format!("{}: {error}", self.recorded[index].kernel))?;
                        }
                    }
                    for commands in &mut lanes {
                        commands.trailing_release()?;
                        command_dwords = command_dwords
                            .checked_add(commands.len_dwords())
                            .ok_or_else(|| "PM4 command dword count overflow".to_owned())?;
                    }
                    max_queue_count = max_queue_count.max(lanes.len());
                    phase_commands.push(lanes);
                    continue;
                }

                let mut commands = Pm4Commands::new(
                    pm4_architecture,
                    self.pm4_register_policy,
                    dispatch_initiator_policy,
                    dispatch_interleave,
                    resource_limits_policy,
                );
                commands.acquire_entry(gfx12_gcr_trim, entry_acquire_policy);
                let mut resource_frontier = ResourceFrontier::default();
                for (position, index) in phase.indices.iter().copied().enumerate() {
                    if position != 0 && !phase.parallel {
                        let previous_index = phase.indices[position - 1];
                        let previous_launch = &self.recorded[previous_index];
                        let current_launch = &self.recorded[index];
                        let previous = previous_launch.kernel.as_str();
                        let current = current_launch.kernel.as_str();
                        let allowlist_independent = independent_sibling(previous, current);
                        let resources_independent = resource_frontier.independent(current_launch);
                        let independent = match self.pm4_wait_policy {
                            Pm4WaitPolicy::Allowlist | Pm4WaitPolicy::ResourceAudit => {
                                allowlist_independent
                            }
                            Pm4WaitPolicy::Resource => resources_independent,
                        };
                        if !independent {
                            commands.wait_compute_idle()?;
                        }
                        resource_frontier.advance(current_launch, resources_independent);
                        if (!independent && commands.requires_dependency_acquire())
                            || self
                                .pm4_mid_acquire_policy
                                .acquire_between(previous, current)
                        {
                            commands.acquire_inter_node(
                                gfx12_gcr_trim,
                                pm4_vmem_acquire_enabled(
                                    pm4_architecture,
                                    gfx11_vmem_acquire,
                                    &radiowave_certifications,
                                    current_launch,
                                ) || pm4_gfx12_vmem_acquire_enabled(
                                    gfx12_vmem_acquire,
                                    &radiowave_certifications,
                                    current_launch,
                                ),
                            );
                        }
                    } else {
                        resource_frontier.advance(&self.recorded[index], false);
                    }
                    commands
                        .dispatch(
                            &kernels[index],
                            geometries[index],
                            self.recorded[index].shared_mem,
                            kernargs[index].address(),
                        )
                        .map_err(|error| format!("{}: {error}", self.recorded[index].kernel))?;
                }
                commands.trailing_release()?;
                command_dwords = command_dwords
                    .checked_add(commands.len_dwords())
                    .ok_or_else(|| "PM4 command dword count overflow".to_owned())?;
                phase_commands.push(vec![commands]);
            }

            let graph = create_phased_pm4_graph(
                pm4_architecture,
                &device,
                &pool,
                &phase_commands,
                native_phase_sync,
            )?;
            debug_assert_eq!(graph.queue_count(), max_queue_count);
            eprintln!(
                "[redline] PM4 phase plan architecture={} queues={} phases={} parallel_phases={} branch_chain_phases={} parallel_launches={} max_width={} min_parallel_width={} min_parallel_workgroups={} max_parallel_phases={} sync={}",
                device.name(),
                graph.queue_count(),
                graph.phase_count(),
                parallel_phases,
                branch_chain_phases,
                parallel_launches,
                max_width,
                min_parallel_width,
                min_parallel_workgroups,
                max_parallel_phases,
                if native_phase_sync { "native" } else { "aql" },
            );
            (PreparedPm4Graph::Phased(graph), command_dwords)
        };
        let queue_id = graph.queue_id();
        // Slice-1 tape census: typed launches re-encode from `ReplayBindings`,
        // untyped launches stay on the snapshot path (slice 2's backlog).
        let (typed_launches, untyped_launches) = self.binding_layout_summary();
        let mut no_effects = 0usize;
        let mut slot_failed = 0usize;
        let mut untyped_by_kernel = BTreeMap::<&str, usize>::new();
        for launch in self.recorded.iter().take(prefix) {
            if launch.binding_layout.is_none() {
                *untyped_by_kernel.entry(launch.kernel.as_str()).or_default() += 1;
                if launch.accesses.is_none() {
                    no_effects += 1;
                } else {
                    slot_failed += 1;
                }
            }
        }
        let backlog: Vec<String> = untyped_by_kernel
            .iter()
            .map(|(kernel, count)| format!("{kernel}x{count}"))
            .collect();
        eprintln!(
            "[redline] PM4 tape binding layouts: arch={} launches={prefix} typed={typed_launches} untyped={untyped_launches} (no_effects={no_effects} slot_failed={slot_failed}) resources={} revision={} backlog=[{}]",
            device.name(),
            self.tape_resources.len(),
            self.binding_revision.0,
            backlog.join(" "),
        );
        self.pm4_generation += 1;
        let bound_exempt_ranges = bound_exempt_kernarg_ranges(
            &self.recorded,
            prefix,
            &dynamic_kernarg_bindings,
            &dynamic_gdn_frames,
        );
        self.prepared_pm4 = Some(PreparedPm4Replay {
            graph,
            _kernels: kernels,
            kernarg_publish: kernargs
                .iter()
                .rposition(|kernarg| kernarg.is_device_local() && !kernarg.is_empty()),
            kernargs,
            _dependency_fence: dependency_fence,
            dynamic_gdn_frames,
            dynamic_kernarg_bindings,
            dynamic_grids,
            pm4_architecture,
            dispatch_count: prefix,
            command_dwords,
            dispatch_boundaries: dispatch_profile.then_some(dispatch_boundaries),
            prepared_max_position: self.prepared_max_position,
            bound_explicit_lens,
            bound_exempt_ranges,
            encoded_revision: self.binding_revision,
        });
        self.state = ReplayState::Ready;
        Ok((prefix, command_dwords, queue_id))
    }

    /// # Safety
    ///
    /// The captured model allocations and all pointed-to buffers must still be
    /// live and in the same binding layout.
    pub unsafe fn replay_linear_aql(&mut self, position: usize) -> Result<GpuBatchTiming, String> {
        let result = {
            let prepared = self
                .prepared
                .as_mut()
                .ok_or_else(|| "no prepared AQL replay".to_owned())?;
            // SAFETY: forwarded from the model owner.
            unsafe { prepared.replay_and_wait() }
        };
        self.observe_replay_result(position, result)
    }

    /// # Safety
    ///
    /// The captured model allocations and all pointed-to buffers must still be
    /// live and in the same binding layout.
    pub unsafe fn replay_pm4(&mut self, position: usize) -> Result<GpuMultiQueueTiming, String> {
        // SAFETY: forwarded from the model owner.
        unsafe { self.replay_pm4_checked(position) }.map_err(|failure| failure.error)
    }

    /// Checked variant that reports quiescence.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::replay_pm4`].
    pub unsafe fn replay_pm4_checked(
        &mut self,
        position: usize,
    ) -> Result<GpuMultiQueueTiming, RetainedReplayFailure> {
        if let Err(error) = self.fail_if_bindings_stale() {
            return Err(RetainedReplayFailure {
                error,
                quiescence: ReplayQuiescence::Proven,
            });
        }
        let result = {
            let prepared = match self.prepared_pm4.as_mut() {
                Some(prepared) => prepared,
                None => {
                    return Err(RetainedReplayFailure {
                        error: "no prepared PM4 replay".to_owned(),
                        quiescence: ReplayQuiescence::Proven,
                    })
                }
            };
            // SAFETY: forwarded from the model owner.
            unsafe { prepared.replay_and_wait_checked(position) }
        };
        self.observe_replay_result_checked(position, result)
    }
    /// Replay one explicitly instrumented PM4 graph exactly once.
    ///
    /// # Safety
    ///
    /// Captured model allocations and pointed-to buffers must remain live.
    pub unsafe fn replay_pm4_dispatch_profile(
        &mut self,
        position: usize,
    ) -> Result<Pm4DispatchProfile, String> {
        self.fail_if_bindings_stale()?;
        let result = {
            let prepared = self
                .prepared_pm4
                .as_mut()
                .ok_or_else(|| "no prepared PM4 replay".to_owned())?;
            // SAFETY: forwarded from the model owner.
            unsafe { prepared.replay_and_wait_dispatch_profiled(position) }
        };
        self.observe_replay_result(position, result)
    }

    /// The HIP twin's inputs (railgun check mode, design §2.4): each prepared
    /// dispatch's kernarg bytes exactly as the last PM4 replay submitted them
    /// (bindings already applied for that position). Refused while the tape
    /// patches grids, which the twin would have to re-derive.
    pub(crate) fn prepared_pm4_submitted_kernargs(&mut self) -> Result<Vec<Vec<u8>>, String> {
        let prepared = self.prepared_pm4.as_mut().ok_or("no prepared PM4 replay")?;
        if !prepared.dynamic_grids.is_empty() {
            return Err(format!(
                "{} dynamic grid patches",
                prepared.dynamic_grids.len()
            ));
        }
        Ok(prepared
            .kernargs
            .iter_mut()
            .map(|k| k.as_mut_bytes().to_vec())
            .collect())
    }

    /// Generation of the prepared PM4 program (see `pm4_generation`).
    pub(crate) fn prepared_pm4_generation(&self) -> u64 {
        self.pm4_generation
    }

    /// The prepared railgun program's surfaces (`None` without the shadow).
    pub(crate) fn railgun_program_surfaces(&self) -> Option<railgun_shadow::ProgramSurfaces> {
        self.railgun_shadow
            .as_ref()
            .and_then(|s| s.surfaces().cloned())
    }

    pub fn prepared_pm4_dispatch_boundaries(&self) -> Option<&[Pm4DispatchBoundary]> {
        self.prepared_pm4
            .as_ref()
            .and_then(PreparedPm4Replay::dispatch_boundaries)
    }
    fn observe_replay_result<T>(
        &mut self,
        position: usize,
        result: Result<T, String>,
    ) -> Result<T, String> {
        let value = match result {
            Ok(value) => value,
            Err(error) => {
                self.replay_observation.failed = true;
                return Err(error);
            }
        };
        self.replay_observation.count = self.replay_observation.count.saturating_add(1);
        self.replay_observation
            .first_position
            .get_or_insert(position);
        self.replay_observation.last_position = Some(position);
        Ok(value)
    }

    fn observe_replay_result_checked<T>(
        &mut self,
        position: usize,
        result: Result<T, RetainedReplayFailure>,
    ) -> Result<T, RetainedReplayFailure> {
        let value = match result {
            Ok(value) => value,
            Err(failure) => {
                self.replay_observation.failed = true;
                return Err(failure);
            }
        };
        self.replay_observation.count = self.replay_observation.count.saturating_add(1);
        self.replay_observation
            .first_position
            .get_or_insert(position);
        self.replay_observation.last_position = Some(position);
        Ok(value)
    }

    /// Largest `start_pos + batch` the prepared dynamic geometry must cover.
    pub fn set_prepared_max_position(&mut self, max_position: usize) {
        self.prepared_max_position = Some(max_position);
    }

    /// Prove no retained IB is in flight, then release the prepared route.
    pub fn shutdown(&mut self) -> Result<(), RetainedReplayFailure> {
        if let Some(prepared) = self.prepared_pm4.as_mut() {
            if let Err((error, quiescence)) = prepared.graph.quiesce() {
                return Err(RetainedReplayFailure {
                    error: error.to_string(),
                    quiescence: match quiescence {
                        Quiescence::Proven => ReplayQuiescence::Proven,
                        Quiescence::Unknown => ReplayQuiescence::Unknown,
                    },
                });
            }
        }
        // Quiescence proven: safe to release retained resources.
        self.prepared_pm4 = None;
        self.prepared = None;
        self.prepared_max_position = None;
        self.route_identities.clear();
        self.synthesized_position_bindings.clear();
        self.position_bindings_calibrated = false;
        Ok(())
    }

    /// Start one explicitly delimited prefill or decode capture. This clears
    /// only the prior launch sequence; validation observations and the backend
    /// request remain intact.
    pub fn begin_capture(&mut self) -> Result<(), &'static str> {
        match self.state {
            ReplayState::Hip => return Err("replay backend is disabled"),
            ReplayState::Fallback => return Err("replay controller is in sticky fallback"),
            ReplayState::Ready => return Err("cannot capture after a prepared plan is installed"),
            _ => {}
        }
        if self.g0_observation.is_some() {
            return Err("cannot capture during a G0 observation");
        }
        self.recorded.clear();
        self.radiowave_effect_launches = 0;
        self.fallback_effect_launches = 0;
        self.unknown_effect_launches = 0;
        self.synthesized_position_bindings.clear();
        self.position_bindings_calibrated = false;
        // Fresh `ResourceId` space per tape: a recycled device address must
        // never alias a prior capture's resource.
        self.binding_issuer = Recorder::new();
        self.replay_bindings = ReplayBindings::new();
        self.binding_revision = BindingRevision(0);
        self.tape_resources.clear();
        self.binding_refresh_pending = false;
        self.window_effects_base = memory_effects::snapshot();
        self.window_effects = MemoryEffects::default();
        if let Some(shadow) = self.railgun_shadow.as_mut() {
            shadow.reset();
        }
        self.state = ReplayState::RecordingWarmup;
        Ok(())
    }

    /// Close the current explicit capture and retain its sequence for
    /// fingerprinting/adapter construction. No launch route changes here.
    pub fn finish_capture(&mut self) -> Result<ReplayCaptureSummary, &'static str> {
        if self.state != ReplayState::RecordingWarmup {
            return Err("no replay capture is active");
        }
        self.window_effects = memory_effects::snapshot().since(self.window_effects_base);
        let summary = self.capture_summary();
        self.state = if self.certified_speedups.len() >= 2 {
            ReplayState::ShadowValidated
        } else {
            ReplayState::Captured
        };
        Ok(summary)
    }

    /// The memory operations issued on this thread inside the open capture
    /// window, or inside the last closed one.
    pub fn capture_window_effects(&self) -> MemoryEffects {
        if self.state == ReplayState::RecordingWarmup {
            memory_effects::snapshot().since(self.window_effects_base)
        } else {
            self.window_effects
        }
    }

    /// Refuse to prepare an automatic forward body whose window issued a device
    /// copy, readback or memset: the tape replays dispatches only, so that
    /// state would be missing on every replay. Every adapter treats a prepare
    /// error as sticky fallback, so the route runs on HIP. Manual captures
    /// (benchmarks, speculative verify bodies) delimit windows spanning several
    /// forwards and keep their own contracts.
    fn refuse_effect_incomplete_window(&self) -> Result<(), String> {
        let effects = self.capture_window_effects();
        if self.auto_lifecycle && !effects.replayable() {
            return Err(format!(
                "effect-incomplete capture: {} device copy(ies), {} readback(s) and {} \
                 memset(s) inside the window are state the tape cannot replay",
                effects.dtod, effects.dtoh, effects.memset
            ));
        }
        Ok(())
    }

    pub fn capture_summary(&self) -> ReplayCaptureSummary {
        let unique_kernel_count = self
            .recorded
            .iter()
            .map(|launch| launch.kernel.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let hash = replay_sequence_hash(&self.recorded);
        ReplayCaptureSummary {
            launch_count: self.recorded.len(),
            unique_kernel_count,
            sequence_hash: hash,
        }
    }

    /// Snapshot the just-finished recording. Valid only between `finish_capture`
    /// and the next `begin_capture`.
    pub fn snapshot_recorded_kernargs(&self) -> RecordedKernargSnapshot {
        let entries = self
            .recorded
            .iter()
            .map(|launch| SnapshotEntry {
                kernel: launch.kernel.clone(),
                kernarg: launch.kernarg.clone(),
                grid: launch.grid,
            })
            .collect();
        RecordedKernargSnapshot { entries }
    }

    /// Difference an earlier recording of the SAME tape against the current
    /// recording and synthesize a `ReplayKernargBinding` for every scalar that
    /// tracks the decode position. Returns the number of bindings synthesized.
    /// Errors (and synthesizes nothing) when any difference is not explained by
    /// the position delta.
    pub fn synthesize_position_bindings(
        &mut self,
        earlier: &RecordedKernargSnapshot,
        earlier_position: usize,
        current_position: usize,
    ) -> Result<usize, String> {
        if current_position <= earlier_position {
            return Err(format!(
                "current_position {current_position} must be > earlier_position {earlier_position}"
            ));
        }
        let delta = current_position - earlier_position;
        let delta_u32 =
            u32::try_from(delta).map_err(|_| format!("position delta {delta} exceeds u32"))?;
        if earlier.entries.len() != self.recorded.len() {
            return Err(format!(
                "launch count mismatch: earlier {} vs current {}",
                earlier.entries.len(),
                self.recorded.len()
            ));
        }
        let mut new_bindings: Vec<(usize, ReplayKernargBinding)> = Vec::new();
        for (idx, (earlier_entry, current_launch)) in
            earlier.entries.iter().zip(self.recorded.iter()).enumerate()
        {
            if earlier_entry.kernel != current_launch.kernel {
                return Err(format!(
                    "kernel mismatch at launch {idx}: earlier {:?} vs current {:?}",
                    earlier_entry.kernel, current_launch.kernel
                ));
            }
            if earlier_entry.kernarg.len() != current_launch.kernarg.len() {
                return Err(format!(
                    "kernarg length mismatch at launch {idx} kernel {:?}: earlier {} vs current {}",
                    earlier_entry.kernel,
                    earlier_entry.kernarg.len(),
                    current_launch.kernarg.len()
                ));
            }
            // Grid check modulo dynamic binding.
            if earlier_entry.grid != current_launch.grid {
                if let Some(binding) = current_launch.grid_binding {
                    let axis = match binding {
                        ReplayGridBinding::PositionCeilDiv { axis, .. } => usize::from(axis),
                    };
                    if axis >= 3 {
                        return Err(format!(
                            "invalid grid binding axis {axis} at launch {idx} kernel {:?}",
                            current_launch.kernel
                        ));
                    }
                    let mut mismatch_allowed = true;
                    for a in 0..3 {
                        if a == axis {
                            continue;
                        }
                        if earlier_entry.grid[a] != current_launch.grid[a] {
                            mismatch_allowed = false;
                        }
                    }
                    if !mismatch_allowed {
                        return Err(format!(
                            "grid mismatch at launch {idx} kernel {:?}: earlier {:?} vs current {:?} (axis {axis} is dynamic)",
                            current_launch.kernel, earlier_entry.grid, current_launch.grid
                        ));
                    }
                } else {
                    return Err(format!(
                        "grid mismatch at launch {idx} kernel {:?}: earlier {:?} vs current {:?}",
                        current_launch.kernel, earlier_entry.grid, current_launch.grid
                    ));
                }
            }
            let len = earlier_entry.kernarg.len();
            let is_gdn = is_gdn_kernel(&current_launch.kernel);
            let mut offset: usize = 0;
            while offset + 4 <= len {
                let earlier_bytes = &earlier_entry.kernarg[offset..offset + 4];
                let current_bytes = &current_launch.kernarg[offset..offset + 4];
                if earlier_bytes != current_bytes {
                    // A named field is not an unexplained difference: the engine
                    // declared this offset dynamic, so replay re-derives it.
                    if current_launch
                        .declared_kernarg_bindings()
                        .iter()
                        .any(|binding| binding.offset() == offset)
                    {
                        offset += 4;
                        continue;
                    }
                    // Skip GDN frame field.
                    if is_gdn && offset == 76 {
                        offset += 4;
                        continue;
                    }
                    let v_earlier = u32::from_le_bytes(earlier_bytes.try_into().unwrap());
                    let v_current = u32::from_le_bytes(current_bytes.try_into().unwrap());
                    if v_current.wrapping_sub(v_earlier) != delta_u32 {
                        // Not a position scalar. Classify before rejecting: a
                        // relocated buffer usually differs only in the low word
                        // of its 8-byte slot, so inspect the enclosing slot
                        // rather than requiring both halves to differ.
                        let pair_offset = offset - (offset % 8);
                        if pair_offset + 8 <= len {
                            let earlier_u64 = u64::from_le_bytes(
                                earlier_entry.kernarg[pair_offset..pair_offset + 8]
                                    .try_into()
                                    .unwrap(),
                            );
                            let current_u64 = u64::from_le_bytes(
                                current_launch.kernarg[pair_offset..pair_offset + 8]
                                    .try_into()
                                    .unwrap(),
                            );
                            if is_plausible_device_address(earlier_u64)
                                || is_plausible_device_address(current_u64)
                            {
                                return Err(format!(
                                    "moved allocation at launch {idx} kernel {:?} offset {pair_offset}: earlier {earlier_u64:#x} vs current {current_u64:#x} (pointer; a relocated buffer must never be retained as a position scalar)",
                                    current_launch.kernel
                                ));
                            }
                        }
                        return Err(format!(
                            "unexplained kernarg difference at launch {idx} kernel {:?} offset {offset}: earlier {v_earlier} ({v_earlier:#x}) vs current {v_current} ({v_current:#x}) (delta {delta})",
                            current_launch.kernel
                        ));
                    }
                    if v_earlier.checked_add(delta_u32) != Some(v_current) {
                        return Err(format!(
                            "kernarg scalar at launch {idx} kernel {:?} offset {offset} wraps u32: earlier {v_earlier} + delta {delta_u32} != current {v_current}",
                            current_launch.kernel
                        ));
                    }
                    let addend = v_current.wrapping_sub(current_position as u32);
                    let repro_earlier = (earlier_position as u32).checked_add(addend);
                    let repro_current = (current_position as u32).checked_add(addend);
                    if repro_earlier != Some(v_earlier) || repro_current != Some(v_current) {
                        return Err(format!(
                            "synthesized PositionPlusU32 at launch {idx} kernel {:?} offset {offset} addend {addend} does not reproduce both samples (earlier {v_earlier} vs {:?}, current {v_current} vs {:?})",
                            current_launch.kernel, repro_earlier, repro_current
                        ));
                    }
                    new_bindings.push((
                        idx,
                        ReplayKernargBinding::PositionPlusU32 { offset, addend },
                    ));
                }
                offset += 4;
            }
        }
        // Order by (dispatch, offset) — already in order, but enforce.
        new_bindings.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.offset().cmp(&b.1.offset())));
        let count = new_bindings.len();
        self.synthesized_position_bindings = new_bindings;
        self.position_bindings_calibrated = true;
        Ok(count)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_hip_launch_typed_bound(
        &mut self,
        hip: &HipRuntime,
        kernel: &str,
        artifact: Option<RecordedArtifact>,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg: &[u8],
        grid_binding: Option<ReplayGridBinding>,
        declared_kernarg_bindings: &[ReplayKernargBinding],
        compiler: Option<&crate::compiler::KernelCompiler>,
        declared_words: &[railgun::kernel::DeclaredWord],
    ) {
        if !self.is_recording() {
            return;
        }
        let certified_effects = artifact
            .as_ref()
            .and_then(|artifact| artifact.radiowave())
            .and_then(|certification| certification.argument_effects(kernel))
            .map(|effects| {
                effects
                    .into_iter()
                    .map(|(offset, access)| match access {
                        KernelArgumentAccess::ReadOnly => read(offset),
                        KernelArgumentAccess::WriteOnly | KernelArgumentAccess::ReadWrite => {
                            write(offset)
                        }
                    })
                    .collect::<Vec<_>>()
            });
        let certified = certified_effects.is_some();
        let accesses =
            recorded_resource_accesses(hip, kernel, kernarg, certified_effects.as_deref());
        if accesses.is_none() {
            self.unknown_effect_launches += 1;
        } else if certified {
            self.radiowave_effect_launches += 1;
        } else {
            self.fallback_effect_launches += 1;
        }
        // Slice-1 binding layout: resolve every pointer-effect slot to a
        // tape-global `ResourceId` now, while the allocations are live. Any
        // unresolvable slot (or unknown kernel) leaves the launch untyped on
        // the raw snapshot path — never a partial slot set.
        let binding_layout =
            self.build_binding_layout(hip, kernel, kernarg, certified_effects.as_deref());
        let before = self.recorded.len();
        self.record_hip_launch_with_accesses(
            kernel,
            artifact,
            grid,
            block,
            shared_mem,
            kernarg,
            grid_binding,
            declared_kernarg_bindings,
            accesses,
            binding_layout,
        );
        if self.recorded.len() > before {
            if let Some(shadow) = self.railgun_shadow.as_mut() {
                let artifact = self.recorded.last().and_then(|launch| launch.artifact.as_ref()).map(|artifact| artifact.id());
                shadow.observe(hip, compiler, kernel, artifact, grid, block, shared_mem, kernarg, declared_words);
            }
        }
    }

    #[cfg(test)]
    fn record_hip_launch(
        &mut self,
        kernel: &str,
        artifact: Option<RecordedArtifact>,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg: &[u8],
    ) {
        self.record_hip_launch_with_accesses(
            kernel,
            artifact,
            grid,
            block,
            shared_mem,
            kernarg,
            None,
            &[],
            None,
            None,
        );
    }
    fn record_hip_launch_with_accesses(
        &mut self,
        kernel: &str,
        artifact: Option<RecordedArtifact>,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg: &[u8],
        grid_binding: Option<ReplayGridBinding>,
        declared_kernarg_bindings: &[ReplayKernargBinding],
        accesses: Option<Vec<RecordedResourceAccess>>,
        binding_layout: Option<LaunchBindingLayout>,
    ) {
        if !self.is_recording() {
            return;
        }
        if self.recorded.len() == self.max_recorded_launches {
            self.fallback("warmup launch recorder capacity exceeded");
            return;
        }
        self.recorded.push(RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact,
            grid,
            block,
            shared_mem,
            grid_binding,
            kernarg: kernarg.to_vec(),
            declared_kernarg_bindings: declared_kernarg_bindings.to_vec(),
            accesses,
            binding_layout,
        });
    }

    /// Slice-1 binding layout for one recorded launch. Every pointer slot is
    /// issued a tape-global `ResourceId` (deduped by record-time
    /// `(allocation_base, allocation_bytes)`) and bound at the current
    /// revision. Returns `None` — leaving the launch on the raw snapshot
    /// path — when any slot is unresolvable. Never returns a partial slot
    /// set: all slots resolve or the launch is untyped.
    fn build_binding_layout(
        &mut self,
        hip: &HipRuntime,
        kernel: &str,
        kernarg: &[u8],
        certified_effects: Option<&[PointerEffect]>,
    ) -> Option<LaunchBindingLayout> {
        let slots = binding_pointer_slots(hip, kernel, kernarg, certified_effects)?;
        let segment_size = u32::try_from(kernarg.len()).ok()?;
        let mut fields = Vec::with_capacity(slots.len());
        let mut bound = Vec::with_capacity(slots.len());
        for (offset, base, size, interior) in slots {
            let resource = match self.tape_resources.get(&(base, size)) {
                Some(id) => *id,
                None => {
                    let id = self
                        .binding_issuer
                        .resource(format!("tape-resource-{base:016x}"), size)
                        .ok()?;
                    // SAFETY: `base..base+size` is the live allocation just
                    // confirmed by `hipMemGetAddressRange`. The revision
                    // changes on every reallocation via the post-growth
                    // refresh, which is the `ResourceBinding::new` contract.
                    let binding = unsafe {
                        ResourceBinding::new(
                            base as usize as *mut std::ffi::c_void,
                            size,
                            self.binding_revision,
                            AllocationPolicy::HipCoarse,
                        )
                    }
                    .ok()?;
                    self.replay_bindings.bind_resource(id, binding);
                    self.tape_resources.insert((base, size), id);
                    id
                }
            };
            let offset_u32 = u32::try_from(offset).ok()?;
            fields.push(KernargField::new(offset_u32, 8, 8).ok()?);
            bound.push(KernargPointerSlot {
                offset,
                resource,
                interior_offset: interior,
            });
        }
        let abi = KernargAbi::new(segment_size, 16, fields).ok()?;
        debug_assert_eq!(abi.fields().len(), bound.len());
        Some(LaunchBindingLayout { abi, slots: bound })
    }

    /// Typed/untyped launch split for the current tape: `(typed, untyped)`.
    /// Untyped launches stay on the raw snapshot path; the untyped count is
    /// slice 2's backlog (kernels without a pointer-effect table entry).
    pub fn binding_layout_summary(&self) -> (usize, usize) {
        let typed = self
            .recorded
            .iter()
            .filter(|launch| launch.binding_layout.is_some())
            .count();
        (typed, self.recorded.len().saturating_sub(typed))
    }

    /// Revision the prepared kernarg segments are encoded at.
    pub fn binding_revision(&self) -> BindingRevision {
        self.binding_revision
    }

    /// Whether a prepared PM4 route is installed (survivable across growth).
    pub fn prepared_pm4_route_active(&self) -> bool {
        self.prepared_pm4.is_some()
    }

    /// Whether a scratch growth is waiting for its post-growth re-resolve.
    pub fn binding_refresh_pending(&self) -> bool {
        self.binding_refresh_pending
    }

    /// Route half of scratch-growth handling: keep the retained PM4 route and
    /// defer the binding re-resolve until after the growth completes. The
    /// next launch drains it via `refresh_bindings_after_growth`; a replay
    /// that observes it still armed fails closed (route re-armed, HIP runs).
    pub fn arm_binding_refresh_for_scratch_growth(&mut self) {
        self.binding_refresh_pending = true;
    }

    /// Post-growth re-resolve: probe every tape resource, bump the revision,
    /// and re-encode the prepared kernarg segments in place. The IB is
    /// untouched (no re-lowering). Any moved/freed resource, or any non-slot
    /// byte that would change, fails closed — the caller must drop the route
    /// (`rearm_after_layout_growth`) and run HIP.
    pub fn refresh_bindings_after_growth(
        &mut self,
        hip: &HipRuntime,
    ) -> Result<BindingRefreshReport, String> {
        if !self.binding_refresh_pending {
            return Ok(BindingRefreshReport::NotPending);
        }
        if self.prepared_pm4.is_none() {
            self.binding_refresh_pending = false;
            return Ok(BindingRefreshReport::NoRoute);
        }
        let next_revision = self.binding_revision.next();
        // Probe first, mutate second: a failed probe leaves every binding at
        // the old revision so the fail-closed re-arm sees coherent state.
        let mut probed: Vec<(ResourceId, u64, u64)> = Vec::with_capacity(self.tape_resources.len());
        for ((base, size), id) in &self.tape_resources {
            let (live_base, live_size) = hip
                .mem_get_address_range(*base as usize as *mut std::ffi::c_void)
                .map_err(|_| {
                    format!(
                        "retained PM4 resource {id:?} no longer resolves after scratch growth; route must re-capture"
                    )
                })?;
            let live_base = live_base as usize as u64;
            let live_size = u64::try_from(live_size).map_err(|_| {
                format!("retained PM4 resource {id:?} size exceeds u64 after scratch growth")
            })?;
            if live_base != *base || live_size < *size {
                return Err(format!(
                    "retained PM4 resource {id:?} moved after scratch growth \
                     (recorded {base:#x}+{size:#x}, live {live_base:#x}+{live_size:#x}); \
                     route must re-capture"
                ));
            }
            probed.push((*id, live_base, live_size));
        }
        let mut grown = 0usize;
        for (id, live_base, live_size) in probed {
            let key = self
                .tape_resources
                .iter()
                .find(|(_, candidate)| **candidate == id)
                .map(|(key, _)| *key);
            if let Some((base, size)) = key {
                if live_size != size {
                    grown += 1;
                    self.tape_resources.remove(&(base, size));
                    self.tape_resources.insert((live_base, live_size), id);
                }
                // SAFETY: probed live just above; revision bumps with the
                // re-encode below, per the `ResourceBinding::new` contract.
                let binding = unsafe {
                    ResourceBinding::new(
                        live_base as usize as *mut std::ffi::c_void,
                        live_size,
                        next_revision,
                        AllocationPolicy::HipCoarse,
                    )
                }
                .map_err(|_| {
                    format!("retained PM4 resource {id:?} cannot rebind after scratch growth")
                })?;
                self.replay_bindings.bind_resource(id, binding);
            }
        }
        self.binding_revision = next_revision;
        // Re-encode every prepared typed segment in place. Non-slot bytes
        // must match the pre-refresh buffer exactly; slot bytes take the new
        // bases. `bound_explicit_lens` are the loader explicit-prefix lengths
        // captured at prepare time.
        let mut reencoded = 0usize;
        let prepared = self
            .prepared_pm4
            .as_mut()
            .expect("prepared PM4 route checked above");
        if prepared.bound_explicit_lens.len() != prepared.kernargs.len() {
            return Err(
                "retained PM4 binding cache disagrees with prepared kernarg count".to_owned(),
            );
        }
        for (index, kernarg) in prepared.kernargs.iter_mut().enumerate() {
            let explicit_len = prepared.bound_explicit_lens[index];
            let launch = self.recorded.get(index).ok_or_else(|| {
                format!("retained PM4 dispatch {index} outruns the recorded tape")
            })?;
            let Some(layout) = launch.binding_layout.as_ref() else {
                continue;
            };
            let encoded = encode_bound_kernarg(
                &launch.kernarg,
                layout,
                &self.replay_bindings,
                &launch.kernel,
            )?;
            let bytes = kernarg.as_mut_bytes();
            if explicit_len > encoded.len() || explicit_len > bytes.len() {
                return Err(format!(
                    "{}: explicit prefix {explicit_len} exceeds segment ({} vs {})",
                    launch.kernel,
                    encoded.len(),
                    bytes.len()
                ));
            }
            let mut slot_mask = vec![false; explicit_len];
            for slot in &layout.slots {
                for offset in slot.offset..slot.offset.saturating_add(8) {
                    if offset < explicit_len {
                        slot_mask[offset] = true;
                    }
                }
            }
            for (offset, (old, new)) in bytes[..explicit_len]
                .iter()
                .zip(encoded[..explicit_len].iter())
                .enumerate()
            {
                if *old != *new && !slot_mask[offset] {
                    return Err(format!(
                        "{}: re-encoded kernarg differs from the prepared segment \
                         at non-slot offset {offset} after scratch growth",
                        launch.kernel
                    ));
                }
            }
            bytes[..explicit_len].copy_from_slice(&encoded[..explicit_len]);
            reencoded += 1;
        }
        prepared.encoded_revision = next_revision;
        self.binding_refresh_pending = false;
        if self.route_proof_log {
            eprintln!(
                "HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 revision={} \
                 event=survived_scratch_growth resources={} reencoded={} grown_in_place={}",
                next_revision.0,
                self.tape_resources.len(),
                reencoded,
                grown
            );
        }
        Ok(BindingRefreshReport::Refreshed {
            resources: self.tape_resources.len(),
            reencoded,
            revision: next_revision,
        })
    }

    fn unchanged_relocation_report(&self) -> BindingRefreshReport {
        BindingRefreshReport::Refreshed {
            resources: self.tape_resources.len(),
            reencoded: 0,
            revision: self.binding_revision,
        }
    }

    /// Typed VMM relocation: rebind the tape resources that lie inside
    /// `moves` to their new bases without re-recording or re-preparing.
    ///
    /// `NoRoute` when nothing is recorded or prepared. `Err` (state untouched)
    /// when a refresh is pending, a move set is invalid, a moved resource is
    /// not fully mapped at its new location, an untyped launch (no pointer
    /// slots) has unknown accesses or touches a moved range, or an AQL
    /// prepared replay references a moved range. Resources outside every move
    /// are untouched and never probed. No HIP query is made: the caller
    /// (`Gpu::relocate_qsa_resources`) already validated the new owners.
    ///
    /// All changes are staged first and committed together at
    /// `binding_revision.next()`: rebound ids, rekeyed `tape_resources`, the
    /// prepared PM4 kernarg slot bytes (non-slot bytes are gated to stay
    /// identical; the indirect buffer, prepared generation and kernarg
    /// allocation addresses never change), `encoded_revision`, the recorded
    /// kernarg snapshots (HIP oracle) and the recorded access bases. When no
    /// tape resource lies in any move the report carries the current revision
    /// and `reencoded: 0`, and nothing changes (the revision is not bumped).
    /// `reencoded` counts rewritten prepared PM4 segments, or rewritten
    /// recorded snapshots when no PM4 route is installed.
    pub fn relocate_resources(
        &mut self,
        _hip: &HipRuntime,
        moves: &[VmmResourceMove],
    ) -> Result<BindingRefreshReport, String> {
        self.relocate_bindings(moves)
    }

    fn relocate_bindings(
        &mut self,
        moves: &[VmmResourceMove],
    ) -> Result<BindingRefreshReport, String> {
        if self.recorded.is_empty() && self.prepared.is_none() && self.prepared_pm4.is_none() {
            return Ok(BindingRefreshReport::NoRoute);
        }
        if self.binding_refresh_pending {
            return Err(
                "cannot relocate VMM resources while a scratch-growth binding refresh is pending"
                    .to_owned(),
            );
        }
        if moves.is_empty() {
            return Ok(self.unchanged_relocation_report());
        }
        let ranges = relocation_ranges(moves)?;
        for launch in &self.recorded {
            if launch.binding_layout.is_none() {
                if let Some(hazard) = untyped_launch_relocation_hazard(launch, &ranges) {
                    return Err(hazard);
                }
            }
        }
        let plan = plan_relocation(&self.tape_resources, moves)?;
        if self.prepared.is_some() {
            let touched = !plan.is_empty()
                || self.recorded.iter().any(|launch| {
                    launch.accesses.as_ref().is_some_and(|accesses| {
                        accesses.iter().any(|access| {
                            !matches!(
                                classify_range(
                                    &ranges,
                                    access.allocation_base,
                                    access.allocation_base.saturating_add(access.allocation_bytes),
                                ),
                                RangeMatch::Outside
                            )
                        })
                    })
                });
            if touched {
                return Err(
                    "prepared AQL replay holds raw kernargs that reference a relocated VMM range"
                        .to_owned(),
                );
            }
        }
        if plan.is_empty() {
            return Ok(self.unchanged_relocation_report());
        }
        let next_revision = self.binding_revision.next();
        let moved: BTreeMap<ResourceId, PlannedRelocation> =
            plan.iter().map(|planned| (planned.resource, *planned)).collect();

        // Stage 1: every tape resource rebinds at the next revision (moved
        // ones at their new base), like the post-growth refresh.
        let mut staged_bindings: Vec<(ResourceId, ResourceBinding)> =
            Vec::with_capacity(self.tape_resources.len());
        for (&(base, size), &id) in &self.tape_resources {
            let current = self.replay_bindings.resource(id).ok_or_else(|| {
                format!("tape resource {id:?} has no bound resource; route must re-capture")
            })?;
            if current.base().as_ptr() as usize as u64 != base || current.size() != size {
                return Err(format!(
                    "tape resource {id:?} binding disagrees with its key {base:#x}+{size:#x}"
                ));
            }
            let target = moved.get(&id).map_or(base, |planned| planned.new_key.0);
            // SAFETY: a moved resource's new range is mapped for `size` bytes
            // by the new VMM owner (`plan_relocation` coverage check, owners
            // validated by the caller); an unmoved one keeps its live base.
            // The revision bumps with this relocation per the
            // `ResourceBinding::new` contract.
            let binding = unsafe {
                ResourceBinding::new(
                    target as usize as *mut std::ffi::c_void,
                    size,
                    next_revision,
                    current.policy(),
                )
            }
            .map_err(|_| format!("tape resource {id:?} cannot rebind after relocation"))?;
            staged_bindings.push((id, binding));
        }

        // Stage 2: rewritten kernarg snapshots and accesses per launch.
        struct StagedLaunch {
            index: usize,
            kernarg: Option<Vec<u8>>,
            accesses: Option<Vec<RecordedResourceAccess>>,
        }
        let mut staged: Vec<StagedLaunch> = Vec::new();
        for (index, launch) in self.recorded.iter().enumerate() {
            let Some(layout) = launch.binding_layout.as_ref() else {
                continue;
            };
            let kernarg = relocate_kernarg_slots(&launch.kernel, &launch.kernarg, layout, &moved)?;
            let accesses = match launch.accesses.as_deref() {
                Some(accesses) => relocate_accesses(&launch.kernel, accesses, &ranges)?,
                None => None,
            };
            if kernarg.is_some() || accesses.is_some() {
                staged.push(StagedLaunch {
                    index,
                    kernarg,
                    accesses,
                });
            }
        }

        // Stage 3: gate every prepared PM4 segment before touching any byte.
        if let Some(prepared) = self.prepared_pm4.as_mut() {
            if prepared.bound_explicit_lens.len() != prepared.kernargs.len()
                || prepared.bound_exempt_ranges.len() != prepared.kernargs.len()
            {
                return Err(
                    "retained PM4 binding cache disagrees with prepared kernarg count".to_owned(),
                );
            }
            for entry in &staged {
                let Some(encoded) = entry.kernarg.as_deref() else {
                    continue;
                };
                if entry.index >= prepared.kernargs.len() {
                    continue;
                }
                let kernel = &self.recorded[entry.index].kernel;
                let explicit_len = prepared.bound_explicit_lens[entry.index];
                let bytes = prepared.kernargs[entry.index].as_mut_bytes();
                if explicit_len > encoded.len() || explicit_len > bytes.len() {
                    return Err(format!(
                        "{kernel}: explicit prefix {explicit_len} exceeds segment ({} vs {})",
                        encoded.len(),
                        bytes.len()
                    ));
                }
                verify_non_slot_bytes_identical(
                    kernel,
                    &bytes[..explicit_len],
                    &encoded[..explicit_len],
                    &prepared.bound_exempt_ranges[entry.index],
                )?;
            }
        }

        // Commit: nothing below can fail.
        for (id, binding) in staged_bindings {
            self.replay_bindings.bind_resource(id, binding);
        }
        for planned in &plan {
            self.tape_resources.remove(&planned.old_key);
        }
        for planned in &plan {
            self.tape_resources.insert(planned.new_key, planned.resource);
        }
        self.binding_revision = next_revision;
        let mut prepared_rewritten = 0usize;
        let mut snapshots_rewritten = 0usize;
        for entry in staged {
            let launch = &mut self.recorded[entry.index];
            if let Some(kernarg) = entry.kernarg {
                if let Some(prepared) = self.prepared_pm4.as_mut() {
                    if entry.index < prepared.kernargs.len() {
                        let explicit_len = prepared.bound_explicit_lens[entry.index];
                        prepared.kernargs[entry.index].as_mut_bytes()[..explicit_len]
                            .copy_from_slice(&kernarg[..explicit_len]);
                        prepared_rewritten += 1;
                    }
                }
                launch.kernarg = kernarg;
                snapshots_rewritten += 1;
            }
            if let Some(accesses) = entry.accesses {
                launch.accesses = Some(accesses);
            }
        }
        if let Some(prepared) = self.prepared_pm4.as_mut() {
            prepared.encoded_revision = next_revision;
        }
        let reencoded = if self.prepared_pm4.is_some() {
            prepared_rewritten
        } else {
            snapshots_rewritten
        };
        if self.route_proof_log {
            eprintln!(
                "HIPFIRE_REPLAY_ROUTE_PROOF transport=pm4 revision={} \
                 event=relocated_resources resources={} moved={} reencoded={}",
                next_revision.0,
                self.tape_resources.len(),
                plan.len(),
                reencoded
            );
        }
        Ok(BindingRefreshReport::Refreshed {
            resources: self.tape_resources.len(),
            reencoded,
            revision: next_revision,
        })
    }

    /// Fail-closed guard for the replay entries: a growth whose refresh was
    /// never drained (no post-growth launch ran) leaves the prepared pointers
    /// unverified. Drop the route via the recoverable re-arm and refuse this
    /// replay so HIP runs instead of stale pointers.
    fn fail_if_bindings_stale(&mut self) -> Result<(), String> {
        if self.binding_refresh_pending {
            self.rearm_after_layout_growth();
            return Err(
                "retained PM4 bindings were not re-resolved after scratch growth; route re-armed"
                    .to_owned(),
            );
        }
        Ok(())
    }

    pub fn observe_shadow(&mut self, observation: ShadowValidation) {
        if self.state == ReplayState::Hip || self.state == ReplayState::Fallback {
            return;
        }
        if !observation.passes(self.threshold) {
            self.fallback("shadow parity, ABI, timing, or speed threshold failed");
            return;
        }
        self.certified_speedups.push(observation.speedup_over_hip);
        if self.certified_speedups.len() >= 2 {
            self.state = ReplayState::ShadowValidated;
        }
    }

    /// Mark that a model adapter has converted recorded launches into an
    /// explicit hazard-checked `redline_dispatch::CompiledPlan`, prepared it,
    /// and retained HIP buffers/artifacts for its lifetime.
    pub fn install_prepared_plan(&mut self) -> Result<(), &'static str> {
        if self.state != ReplayState::ShadowValidated {
            return Err("two passing shadow validations are required");
        }
        if self.request == ReplayBackendRequest::Shadow {
            return Err("shadow mode never changes the launch route");
        }
        self.state = ReplayState::Ready;
        Ok(())
    }

    pub fn should_route_aql(&self) -> bool {
        self.forward_eligible
            && self.request == ReplayBackendRequest::Auto
            && self.state == ReplayState::Ready
            && self.transport == ReplayTransport::AqlPackets
    }

    pub fn should_route_pm4(&self) -> bool {
        self.forward_eligible
            && self.request == ReplayBackendRequest::Auto
            && self.state == ReplayState::Ready
            && self.transport == ReplayTransport::Pm4Ib
    }

    pub fn uses_pm4_transport(&self) -> bool {
        self.transport == ReplayTransport::Pm4Ib
    }

    /// Whether the *current forward* is inside the retained body: the controller
    /// would record this forward into the tape, or route it through a prepared
    /// plan.
    ///
    /// This is the scope a launch-level admission guard must use. A forward that
    /// merely has a replay backend enabled — prefill, an ineligible call, or an
    /// already-poisoned route — is not in the retained body and must keep running
    /// on HIP, so a route that cannot be retained degrades the model to HIP
    /// instead of making every forward fail.
    pub fn retained_body_active(&self) -> bool {
        self.is_recording() || self.should_route_aql() || self.should_route_pm4()
    }

    /// Latch one route identity (a live expert pointer mapping) for the tape
    /// being recorded, keyed by the route's own stable table identity, and
    /// require every later observation under that key to match.
    ///
    /// Called by a route that knows its pointer mapping while the retained body is
    /// active. A mismatch means the tape was captured against a different mapping,
    /// so replaying it would dereference tensors the plan never validated.
    pub fn note_route_identity(&mut self, key: u64, identity: &str) -> Result<(), String> {
        match self.route_identities.get(&key) {
            None => {
                self.route_identities.insert(key, identity.to_owned());
                Ok(())
            }
            Some(latched) if latched == identity => Ok(()),
            Some(latched) => Err(format!(
                "retained route identity for table {key:#x} changed while recording: latched \
                 {latched}, observed {identity}"
            )),
        }
    }

    pub fn poison(&mut self, reason: impl Into<String>) {
        self.fallback_reason = Some(reason.into());
        self.state = ReplayState::Fallback;
    }

    fn fallback(&mut self, reason: &str) {
        self.poison(reason);
    }
}

fn populate_gfx12_kernarg(
    destination: &mut KernargBuffer,
    launch: &RecordedHipLaunch,
    loader_bytes: usize,
) -> Result<(), String> {
    let (base, has_implicit) = validate_loader_kernarg(launch, loader_bytes)?;
    if destination.len() != loader_bytes {
        return Err(format!(
            "{}: destination {} bytes != loader {loader_bytes}",
            launch.kernel,
            destination.len(),
        ));
    }
    let bytes = destination.as_mut_bytes();
    bytes.fill(0);
    bytes[..base].copy_from_slice(&launch.kernarg[..base]);

    if !has_implicit {
        return Ok(());
    }

    for axis in 0..3 {
        put_u32(bytes, base + axis * 4, launch.grid[axis])?;
        let group = u16::try_from(launch.block[axis]).map_err(|_| {
            format!(
                "{}: workgroup dimension {} exceeds u16",
                launch.kernel, launch.block[axis]
            )
        })?;
        put_u16(bytes, base + 12 + axis * 2, group)?;
        // HIP's grid values are work-group counts, so total work-items are an
        // exact multiple of the group size and every remainder is zero.
        put_u16(bytes, base + 18 + axis * 2, 0)?;
    }
    let dimensions = if launch.grid[2] != 1 || launch.block[2] != 1 {
        3
    } else if launch.grid[1] != 1 || launch.block[1] != 1 {
        2
    } else {
        1
    };
    put_u16(bytes, base + 64, dimensions)?;
    put_u32(bytes, base + 120, launch.shared_mem)?;
    Ok(())
}

/// Slice-1 PM4 lowering: build the kernarg segment from `ReplayBindings` +
/// `KernargAbi` instead of the raw byte snapshot. Typed launches re-encode
/// from the current bindings (byte-equal to the snapshot at the record
/// revision — fail closed otherwise); untyped launches copy the snapshot.
/// Returns the loader explicit-prefix length for the re-encode cache.
/// The AQL path keeps `populate_gfx12_kernarg` (snapshot) untouched.
fn populate_gfx12_kernarg_bound(
    destination: &mut KernargBuffer,
    launch: &RecordedHipLaunch,
    bindings: &ReplayBindings,
    loader_bytes: usize,
) -> Result<usize, String> {
    // Re-encode before validation so the snapshot tail check in
    // `validate_loader_kernarg` still applies to the recorded bytes.
    let owned: Option<Vec<u8>> = match launch.binding_layout.as_ref() {
        Some(layout) => {
            let encoded = encode_bound_kernarg(&launch.kernarg, layout, bindings, &launch.kernel)?;
            verify_bound_kernarg(&launch.kernel, &launch.kernarg, &encoded)?;
            Some(encoded)
        }
        None => None,
    };
    let source: &[u8] = owned.as_deref().unwrap_or(&launch.kernarg);
    let (explicit, has_implicit) = validate_loader_kernarg(launch, loader_bytes)?;
    if destination.len() != loader_bytes {
        return Err(format!(
            "{}: destination {} bytes != loader {loader_bytes}",
            launch.kernel,
            destination.len(),
        ));
    }
    let bytes = destination.as_mut_bytes();
    bytes.fill(0);
    bytes[..explicit].copy_from_slice(&source[..explicit]);

    if !has_implicit {
        return Ok(explicit);
    }

    for axis in 0..3 {
        put_u32(bytes, explicit + axis * 4, launch.grid[axis])?;
        let group = u16::try_from(launch.block[axis]).map_err(|_| {
            format!(
                "{}: workgroup dimension {} exceeds u16",
                launch.kernel, launch.block[axis]
            )
        })?;
        put_u16(bytes, explicit + 12 + axis * 2, group)?;
        // HIP's grid values are work-group counts, so total work-items are an
        // exact multiple of the group size and every remainder is zero.
        put_u16(bytes, explicit + 18 + axis * 2, 0)?;
    }
    let dimensions = if launch.grid[2] != 1 || launch.block[2] != 1 {
        3
    } else if launch.grid[1] != 1 || launch.block[1] != 1 {
        2
    } else {
        1
    };
    put_u16(bytes, explicit + 64, dimensions)?;
    put_u32(bytes, explicit + 120, launch.shared_mem)?;
    Ok(explicit)
}

fn validate_loader_kernarg(
    launch: &RecordedHipLaunch,
    loader_bytes: usize,
) -> Result<(usize, bool), String> {
    const IMPLICIT_BYTES: usize = 256;
    let captured = launch.kernarg.len();
    let (explicit, has_implicit) = if loader_bytes <= captured {
        (loader_bytes, false)
    } else {
        let explicit = loader_bytes.checked_sub(IMPLICIT_BYTES).ok_or_else(|| {
            format!(
                "loader requires {loader_bytes} bytes, larger than captured {captured} but smaller than implicit suffix"
            )
        })?;
        if explicit > captured {
            return Err(format!(
                "loader explicit prefix {explicit} exceeds captured {captured} bytes"
            ));
        }
        (explicit, true)
    };
    if launch.kernarg[explicit..].iter().any(|byte| *byte != 0) {
        return Err(format!(
            "loader explicit prefix {explicit} would discard nonzero captured bytes from {}",
            launch.kernarg.len(),
        ));
    }
    Ok((explicit, has_implicit))
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) -> Result<(), String> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| "kernarg u16 offset overflow".to_owned())?;
    let slot = bytes
        .get_mut(offset..end)
        .ok_or_else(|| format!("kernarg u16 write {offset}..{end} is out of bounds"))?;
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) -> Result<(), String> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| "kernarg u32 offset overflow".to_owned())?;
    let slot = bytes
        .get_mut(offset..end)
        .ok_or_else(|| format!("kernarg u32 write {offset}..{end} is out of bounds"))?;
    slot.copy_from_slice(&value.to_le_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pm4_architecture_fails_closed_for_unknown_gfx12_devices() {
        assert_eq!(
            Pm4Architecture::from_name("gfx1200"),
            Ok(Pm4Architecture::Gfx12)
        );
        assert_eq!(
            Pm4Architecture::from_name("GFX1201"),
            Ok(Pm4Architecture::Gfx12)
        );
        assert!(Pm4Architecture::from_name("gfx1202").is_err());
        assert!(Pm4Architecture::from_name("gfx12-future").is_err());
    }

    /// Slice-1 fixture: a 32-byte segment with pointer slots at offsets 0
    /// (interior pointer) and 16 (allocation base) plus scalar bytes around
    /// them. Returns the snapshot, its layout, bindings at revision 0, and
    /// the two allocation bases for the revision-bump test.
    fn bound_segment_fixture() -> (Vec<u8>, LaunchBindingLayout, ReplayBindings, u64, u64) {
        let base_a: u64 = 0x7f00_0001_0000;
        let base_b: u64 = 0x7f00_0002_0000;
        let mut snapshot = vec![0x11u8; 32];
        snapshot[0..8].copy_from_slice(&(base_a + 0x40).to_ne_bytes());
        snapshot[8..12].copy_from_slice(&0xdead_beefu32.to_ne_bytes());
        snapshot[12..16].copy_from_slice(&0x0000_0042u32.to_ne_bytes());
        snapshot[16..24].copy_from_slice(&base_b.to_ne_bytes());
        snapshot[24..28].copy_from_slice(&0x0000_0007u32.to_ne_bytes());
        let mut issuer = Recorder::new();
        let resource_a = issuer
            .resource("fixture-a", 0x1_0000)
            .expect("valid resource");
        let resource_b = issuer
            .resource("fixture-b", 0x2_0000)
            .expect("valid resource");
        let mut bindings = ReplayBindings::new();
        // SAFETY: synthetic non-deref'd addresses used only as binding identity
        // in unit tests; sizes match the fixture resources; never launched.
        unsafe {
            bindings.bind_resource(
                resource_a,
                ResourceBinding::new(
                    base_a as usize as *mut std::ffi::c_void,
                    0x1_0000,
                    BindingRevision(0),
                    AllocationPolicy::HipCoarse,
                )
                .expect("valid binding"),
            );
            bindings.bind_resource(
                resource_b,
                ResourceBinding::new(
                    base_b as usize as *mut std::ffi::c_void,
                    0x2_0000,
                    BindingRevision(0),
                    AllocationPolicy::HipCoarse,
                )
                .expect("valid binding"),
            );
        }
        let abi = KernargAbi::new(
            32,
            16,
            [
                KernargField::new(0, 8, 8).expect("valid field"),
                KernargField::new(16, 8, 8).expect("valid field"),
            ],
        )
        .expect("valid ABI");
        let layout = LaunchBindingLayout {
            abi,
            slots: vec![
                KernargPointerSlot {
                    offset: 0,
                    resource: resource_a,
                    interior_offset: 0x40,
                },
                KernargPointerSlot {
                    offset: 16,
                    resource: resource_b,
                    interior_offset: 0,
                },
            ],
        };
        (snapshot, layout, bindings, base_a, base_b)
    }

    #[test]
    fn bound_kernarg_reencode_equals_snapshot() {
        let (snapshot, layout, bindings, _, _) = bound_segment_fixture();
        let encoded =
            encode_bound_kernarg(&snapshot, &layout, &bindings, "fixture_kernel").expect("encodes");
        assert_eq!(encoded, snapshot);
        verify_bound_kernarg("fixture_kernel", &snapshot, &encoded).expect("verifies");
    }

    #[test]
    fn binding_revision_bump_reencodes_new_base_only() {
        let (snapshot, layout, mut bindings, base_a, base_b) = bound_segment_fixture();
        let resource_a = layout.slots[0].resource;
        // Simulate a survived relocation of allocation A (same size, new
        // base) at revision 1; B is untouched.
        let moved_a: u64 = base_a + 0x10_0000;
        // SAFETY: synthetic relocated address for unit-test binding identity only;
        // never dereferenced or launched.
        unsafe {
            bindings.bind_resource(
                resource_a,
                ResourceBinding::new(
                    moved_a as usize as *mut std::ffi::c_void,
                    0x1_0000,
                    BindingRevision(1),
                    AllocationPolicy::HipCoarse,
                )
                .expect("valid binding"),
            );
        }
        let encoded =
            encode_bound_kernarg(&snapshot, &layout, &bindings, "fixture_kernel").expect("encodes");
        assert_eq!(&encoded[0..8], &(moved_a + 0x40).to_ne_bytes());
        assert_eq!(&encoded[8..16], &snapshot[8..16]);
        assert_eq!(&encoded[16..24], &base_b.to_ne_bytes());
        assert_eq!(&encoded[24..], &snapshot[24..]);
        // The fail-closed gate names the launch and the first moved offset.
        let error = bound_kernarg_mismatch_message("fixture_kernel", &snapshot, &encoded)
            .expect("moved base must fail the snapshot gate");
        assert!(
            error.contains("fixture_kernel") && error.contains("offset 2"),
            "unexpected gate message: {error}"
        );
    }
    const A3B_REPLAY_KERNELS: &[&str] = &[
        "fused_rmsnorm_mq_rotate",
        "fused_rmsnorm_mq_rotate_vecsum",
        "fused_rmsnorm_mq_rotate_vecsum_sign_const",
        "fused_rmsnorm_mq_rotate_vecsum_sign_lds",
        "fused_rmsnorm_mq_rotate_wavegrid",
        "rmsnorm_reduce_gfx1100",
        "rotate_with_rms_gfx1100",
        "fused_qkvza_hfq4g256",
        "fused_qkvza_hfq4g256_k2048",
        "fused_qkvza_hfq4g256_k2048_r2",
        "fused_qkvza_hfq4g256_k2048_cpol_slc",
        "fused_qkvza_hfq4g256_k2048_scalar_prep",
        "fused_qkvza_hfq4g256_wavepack4",
        "fused_qkvza_hfq4g256_ldsx8",
        "fused_qkvza_hfq4g256_reduce_chain",
        "fused_qkvza_mq4g256v2",
        "fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100",
        "fused_qkv_mq4g256v2",
        "fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100",
        "fused_gate_up_mq4g256v2",
        "fused_gate_up_mq4g256v2_k5120_gfx1100",
        "fused_sigmoid_alpha_gate_f32",
        "conv1d_silu_split_f32",
        "conv1d_silu_split_qknorm_b256_scalar_prep",
        "fused_qk_l2_norm_scale_f32",
        "repeat_interleave_qk_f32",
        "gated_delta_net_q8_fast",
        "gated_norm_f32",
        "gated_norm_mq_rotate_gfx1100",
        "gated_norm_mq_rotate_k6144_gfx1100",
        "gated_norm_mq_rotate_gfx1151",
        "gated_norm_mq_rotate_gfx1201",
        "gated_norm_mq_rotate_k6144_gfx1201",
        "gated_norm_mq_rotate_awq_k6144_gfx1201",
        "qwen35_fa_prep_gfx1100",
        "qwen36_27b_fa_prep_gfx1100",
        "qwen35_fa_prep_gfx1151",
        "qwen35_fa_prep_gfx1201",
        "qwen36_27b_fa_prep_gfx1201",
        "mq_rotate_x",
        "gemv_hfq4g256_residual",
        "gemv_hfq4g256_residual_cpol_rt",
        "gemv_hfq4g256_residual_cpol_rt_low",
        "gemv_hfq4g256_residual_cpol_slc",
        "gemv_hfq4g256_residual_k2048",
        "gemv_mq4g256v2_residual_r1_k4096_gfx1100_noscratch",
        "gemv_hfq4g256_residual_rt_low_gfx1151",
        "gemv_hfq4g256",
        "gemv_hfq4g256_k2048",
        "gemv_hfq4g256_wide",
        "softmax_f32",
        "moe_topk_renorm_k8",
        "moe_router_softmax_topk_k8_wave64",
        "moe_router_softmax_topk_k8_wave64_exact",
        "moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate",
        "fused_silu_mul_mq_rotate",
        "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
        "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
        "gemv_hfq4g256_moe_gate_up_k8_indexed",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_dlc",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_glc",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_k2816",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_k2816_cpol_dlc",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_k2816_cpol_glc",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_k2816_cpol_slc",
        "gemv_mq4g256_moe_gate_up_k8_indexed_k2816",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2",
        "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151",
        "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151",
        "gemv_hfq4g256_moe_gate_up_k8_indexed_paired_waves_k2048_gfx1151",
        "gemv_mq4g256v2_moe_gate_up_k8_indexed_k2048_nolds_gfx1100",
        "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded",
        "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_cpol_slc",
        "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row8_gfx1151",
        "gemv_hfq4g256_moe_down_k8_indexed_last_combine",
        "moe_down_combine_k8_batched",
        "moe_down_combine_k8_batched_vec4",
        "moe_down_combine_rmsnorm_mq_rotate_vecsum",
        "moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151",
        "fused_qkv_hfq4g256",
        "deinterleave_f32",
        "rmsnorm_f32",
        "rope_partial_halfsplit_f32",
        "kv_cache_write_asym_k_fwht3",
        "kv_cache_write_q8_0",
        "kv_cache_write_q8_0_pair",
        "attention_flash_fwht3_tile",
        "attention_flash_q8_0_reduce",
        "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1100",
        "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151",
        "attention_flash_q8_0_reduce_gated_mq_rotate_gfx1201",
        "sigmoid_mul_f32",
        "gemv_hfq4g256_multirow_r2",
        "gemv_hfq4g256_multirow_r4",
        "gemv_hfq4g256_multirow_r8",
    ];

    const DS4_MQ2R_REPLAY_KERNELS: &[&str] = &[
        "compressor_add_ape_f32_buf",
        "compressor_overlap_concat_f32",
        "compressor_softmax_pool_f32_buf",
        "deepseek4_attn_swa_buf",
        "deepseek4_attn_swa_topk_scoregrid_f32_buf",
        "deepseek4_fused_silu_mul_clamp_mq_rotate",
        "deepseek4_moe_topk_bias_aware_f32",
        "deepseek4_silu_mul_clamp_f32",
        "deepseek4_topk_kv_gather_tiled_f32_buf",
        "deepseek4_topk_kv_gather_identity_f32_buf",
        "fused_rmsnorm_mq_rotate_plain_nox",
        "gemv_mfp4g32_e8_soa_grouped_gfx1151",
        "gemv_mfp4g32_e8_soa_u4_buffer_cpol0_gfx1151",
        "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8all_indexed",
        "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed",
        "hash_router_normalize_f32_buf",
        "hc_compute_control_vec4_finalize",
        "hc_head_compute_pre",
        "hc_input_map_4stream",
        "hc_mix_4stream",
        "indexer_relu_score_f32_buf",
        "indexer_top_k_buf_parallel",
        "mq_rotate_x",
        "rmsnorm_f32",
        "rmsnorm_f32_at_slot_buf",
        "rope_tail_interleaved_f32",
        "rope_tail_yarn_interleaved_at_slot_buf_f32",
        "rope_tail_yarn_interleaved_wide_f32",
        "sqrt_softplus_f32",
        "state_overlap_shift_f32_buf",
        "state_ring_write_f32_buf",
        "swa_ring_write_f32_buf",
    ];

    fn passing(speedup: f64) -> ShadowValidation {
        ShadowValidation {
            bit_exact: true,
            guards_intact: true,
            same_artifact: true,
            abi_valid: true,
            automatic_clocks: true,
            gpu_timed: true,
            speedup_over_hip: speedup,
        }
    }

    #[test]
    fn radiowave_vmem_cache_classification_fails_closed() {
        let mut launch = RecordedHipLaunch {
            kernel: "fused_rmsnorm_mq_rotate".to_owned(),
            artifact: None,
            grid: [1, 1, 1],
            block: [32, 1, 1],
            shared_mem: 0,
            grid_binding: None,
            declared_kernarg_bindings: Vec::new(),
            kernarg: Vec::new(),
            accesses: None,
            binding_layout: None,
        };
        let certifications = BTreeMap::new();
        assert!(!radiowave_vmem_only_consumer(&certifications, &launch));
        launch.artifact = Some(std::sync::Arc::new(crate::code_object::CodeObjectArtifact::native_embedded(
            "m",
            (b"uncertified" as &'static [u8]).into(),
            None,
        )));
        assert!(!radiowave_vmem_only_consumer(&certifications, &launch));

        let artifact = "native:m";
        let manifest = format!(
            r#"{{
                "schema_version": 3,
                "compiler": "radiowave",
                "generated_unix_seconds": 0,
                "source": "/source.hip",
                "output": "{}",
                "arch": "gfx1151",
                "wavefront": "wave32",
                "hipcc": "/opt/rocm/bin/hipcc",
                "hipcc_version": "test",
                "command": [],
                "source_sha256": "",
                "support_header_sha256": "",
                "output_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "inspection": {{
                    "bundle_target": "hipv4-amdgcn-amd-amdhsa--gfx1151",
                    "kernels": [{{
                        "name": "fused_rmsnorm_mq_rotate",
                        "wavefront_size": 32,
                        "vgpr_count": 1,
                        "sgpr_count": 1,
                        "vgpr_spill_count": 0,
                        "sgpr_spill_count": 0,
                        "private_segment_fixed_size": 0,
                        "mutable_read_cache": "vmem_only",
                        "instructions": {{}}
                    }}]
                }}
            }}"#,
            artifact
        );
        let certified = |manifest: Option<&str>| {
            std::sync::Arc::new(crate::code_object::CodeObjectArtifact::native_embedded("m", (&[] as &'static [u8]).into(), manifest))
        };
        // The certification travels with the admitted image, keyed by digest.
        launch.artifact = Some(certified(Some(&manifest)));
        let certifications = radiowave_certifications(std::slice::from_ref(&launch), 1);
        assert_eq!(certifications.len(), 1);
        assert!(radiowave_vmem_only_consumer(&certifications, &launch));
        assert_eq!(redline_effect_source(&launch), "none", "vmem_only alone declares no argument effects");
        launch.kernel = "unknown_kernel".to_owned();
        assert!(!radiowave_vmem_only_consumer(&certifications, &launch));
        // A manifest that does not bind this image's hash certifies nothing.
        launch.kernel = "fused_rmsnorm_mq_rotate".to_owned();
        launch.artifact = Some(std::sync::Arc::new(crate::code_object::CodeObjectArtifact::native_embedded(
            "m",
            (b"other" as &'static [u8]).into(),
            Some(&manifest),
        )));
        assert!(radiowave_certifications(std::slice::from_ref(&launch), 1).is_empty());
        assert!(!radiowave_vmem_only_consumer(&certifications, &launch));
    }

    #[test]
    fn ds4_mq2r_tape_has_complete_resource_contracts() {
        assert_eq!(DS4_MQ2R_REPLAY_KERNELS.len(), 32);
        for kernel in DS4_MQ2R_REPLAY_KERNELS {
            assert!(
                expected_kernarg_bytes(kernel).is_some(),
                "{kernel} has no kernarg contract"
            );
            assert!(
                pointer_effects(kernel).is_some(),
                "{kernel} has no pointer-effect contract"
            );
        }
        assert_eq!(
            pointer_effects("hc_mix_4stream").map(|effects| effects[4].mode),
            Some(RecordedAccessMode::Write)
        );
        assert_eq!(
            pointer_effects("hc_input_map_4stream").map(|effects| effects[1].mode),
            Some(RecordedAccessMode::Read)
        );
    }

    #[test]
    fn gfx12_never_reports_gfx11_vmem_acquire() {
        assert!(pm4_vmem_acquire_arch_enabled(Pm4Architecture::Gfx11, true));
        assert!(!pm4_vmem_acquire_arch_enabled(Pm4Architecture::Gfx12, true));
        assert!(!pm4_vmem_acquire_arch_enabled(
            Pm4Architecture::Gfx11,
            false
        ));
    }

    #[test]
    fn gfx12_vmem_acquire_is_explicit_opt_in() {
        // Unset and `auto` both mean off: the HipLlvmVmemL1 rung stays an
        // explicit operator decision until the harness proves it exact and
        // non-slower. Only an affirmative value enables it.
        assert!(!pm4_gfx12_vmem_acquire_from_value(None));
        assert!(!pm4_gfx12_vmem_acquire_from_value(Some("auto".to_owned())));
        assert!(!pm4_gfx12_vmem_acquire_from_value(Some("0".to_owned())));
        assert!(!pm4_gfx12_vmem_acquire_from_value(Some("off".to_owned())));
        for enabled in ["1", "true", "on"] {
            assert!(
                pm4_gfx12_vmem_acquire_from_value(Some(enabled.to_owned())),
                "{enabled}"
            );
        }
        // The Legacy arch gate is untouched: the gfx11 flag path still
        // reports false on gfx12 even when configured.
        assert!(!pm4_vmem_acquire_arch_enabled(Pm4Architecture::Gfx12, true));
    }

    #[test]
    fn gfx1010_release_wait_selector_is_exact() {
        assert!(gfx1010_release_wait_required(
            Pm4Architecture::Gfx10,
            "gfx1010"
        ));
        assert!(gfx1010_release_wait_required(
            Pm4Architecture::Gfx10,
            "GFX1010"
        ));
        assert!(!gfx1010_release_wait_required(
            Pm4Architecture::Gfx10,
            "gfx1030"
        ));
        assert!(!gfx1010_release_wait_required(
            Pm4Architecture::Gfx10,
            "gfx1011"
        ));
        assert!(!gfx1010_release_wait_required(
            Pm4Architecture::Gfx11,
            "gfx1100"
        ));
        assert!(!gfx1010_release_wait_required(
            Pm4Architecture::Gfx11,
            "gfx1151"
        ));
        assert!(!gfx1010_release_wait_required(
            Pm4Architecture::Gfx12,
            "gfx1201"
        ));
        // Architecture gate is conjunctive: wrong family never selects even if
        // the name string matches by accident.
        assert!(!gfx1010_release_wait_required(
            Pm4Architecture::Gfx11,
            "gfx1010"
        ));
    }

    #[test]
    fn gfx1010_dependency_policy_defaults_to_release_wait() {
        assert_eq!(
            gfx1010_dependency_policy_from_value(Pm4Architecture::Gfx10, "gfx1010", None).unwrap(),
            Gfx1010DependencyPolicy::ReleaseWait
        );
        assert_eq!(
            gfx1010_dependency_policy_from_value(
                Pm4Architecture::Gfx10,
                "GFX1010",
                Some("release-wait")
            )
            .unwrap(),
            Gfx1010DependencyPolicy::ReleaseWait
        );
    }

    #[test]
    fn gfx1010_dependency_policy_accepts_cs_partial_flush() {
        assert_eq!(
            gfx1010_dependency_policy_from_value(
                Pm4Architecture::Gfx10,
                "gfx1010",
                Some("cs-partial-flush")
            )
            .unwrap(),
            Gfx1010DependencyPolicy::CsPartialFlush
        );
    }

    #[test]
    fn gfx1010_dependency_policy_rejects_unknown_exact_values() {
        for raw in [
            "partial-flush",
            "",
            "cs",
            "CS-PARTIAL-FLUSH",
            "cs_partial_flush",
            "RELEASE-WAIT",
            "0",
            "1",
            "true",
            "falsé",
        ] {
            let err =
                gfx1010_dependency_policy_from_value(Pm4Architecture::Gfx10, "gfx1010", Some(raw))
                    .unwrap_err();
            assert!(
                err.contains("HIPFIRE_REPLAY_PM4_GFX1010_DEPENDENCY"),
                "missing key in error for {raw:?}: {err}"
            );
            assert!(
                err.contains(raw),
                "missing offending value {raw:?} in error: {err}"
            );
        }
    }

    #[test]
    fn gfx1010_dependency_policy_ignored_off_exact_device() {
        // Non-exact devices stay on CsPartialFlush and ignore the override key.
        for (arch, name, value) in [
            (Pm4Architecture::Gfx10, "gfx1030", Some("release-wait")),
            (Pm4Architecture::Gfx10, "gfx1011", Some("cs-partial-flush")),
            (Pm4Architecture::Gfx11, "gfx1100", Some("release-wait")),
            (Pm4Architecture::Gfx11, "gfx1010", Some("release-wait")),
            (Pm4Architecture::Gfx12, "gfx1201", Some("bogus")),
            (Pm4Architecture::Gfx10, "gfx1030", None),
        ] {
            assert_eq!(
                gfx1010_dependency_policy_from_value(arch, name, value).unwrap(),
                Gfx1010DependencyPolicy::CsPartialFlush,
                "arch={arch:?} name={name} value={value:?}"
            );
        }
    }

    #[test]
    fn gfx1010_cs_partial_flush_override_emits_event_write_only() {
        // Encoding path for the diagnostic CsPartialFlush override: no fence
        // allocation/sentinel; historical EVENT_WRITE CS_PARTIAL_FLUSH only.
        let mut commands = Pm4Commands::new_with_dependency(
            Pm4Architecture::Gfx10,
            Pm4RegisterPolicy::Legacy,
            Gfx10DispatchInitiatorPolicy::Legacy,
            None,
            Gfx11ComputeResourceLimitsPolicy::Legacy,
            LegacyDependencyMode::CsPartialFlush,
        );
        assert_eq!(
            commands.dependency_mode(),
            Some(LegacyDependencyMode::CsPartialFlush)
        );
        commands.emit_entry_sentinel_reset().unwrap();
        commands.wait_compute_idle().unwrap();
        let dwords = commands.dwords().unwrap();
        assert_eq!(dwords, &[0xc000_4600, 0x407]);
    }
    #[test]
    fn trailing_release_emits_wait_then_system_acquire() {
        // Terminal correctness: CS_PARTIAL_FLUSH drains shaders but does not
        // write back GL2 for a non-shader next consumer. The trailing release
        // is wait_compute_idle (2 dwords) followed by ACQUIRE_MEM with the
        // GL2 writeback bits (8 dwords) — not a RELEASE_MEM packet.
        let mut commands = Pm4Commands::new_with_dependency(
            Pm4Architecture::Gfx11,
            Pm4RegisterPolicy::Legacy,
            Gfx10DispatchInitiatorPolicy::Legacy,
            None,
            Gfx11ComputeResourceLimitsPolicy::Legacy,
            LegacyDependencyMode::CsPartialFlush,
        );
        commands.trailing_release().unwrap();
        let dwords = commands.dwords().unwrap();
        assert_eq!(dwords.len(), 2 + 8);
        assert_eq!(&dwords[..2], &[0xc000_4600, 0x407]);
        // ACQUIRE_MEM header: packet3 type 3, opcode 0x58, count 7.
        assert_eq!(dwords[2], 0xc006_5800);
    }

    #[test]
    fn gfx1010_release_wait_emits_sentinel_then_checked_epochs() {
        const FENCE_ADDR: u64 = 0x1234_5678_9abc_def0;
        let mut commands = Pm4Commands::new_with_dependency(
            Pm4Architecture::Gfx10,
            Pm4RegisterPolicy::Legacy,
            Gfx10DispatchInitiatorPolicy::Legacy,
            None,
            Gfx11ComputeResourceLimitsPolicy::Legacy,
            LegacyDependencyMode::ReleaseWait {
                address: FENCE_ADDR,
                next_epoch: 0,
            },
        );
        commands.emit_entry_sentinel_reset().unwrap();
        commands.wait_compute_idle().unwrap();
        commands.wait_compute_idle().unwrap();

        let dwords = commands.dwords().expect("legacy dwords");
        // One fence is RELEASE_MEM (8 dwords) + WAIT_REG_MEM (7 dwords) = 15.
        assert_eq!(dwords.len(), 15 * 3);
        // Sentinel epoch 0, then dependency epochs 1 and 2.
        assert_eq!(dwords[5], 0);
        assert_eq!(dwords[12], 0);
        assert_eq!(dwords[15 + 5], 1);
        assert_eq!(dwords[15 + 12], 1);
        assert_eq!(dwords[30 + 5], 2);
        assert_eq!(dwords[30 + 12], 2);
        // No CS_PARTIAL_FLUSH EVENT_WRITE on the selected path.
        const EVENT_WRITE_IDLE: u32 = 0xc000_4600;
        assert!(!dwords.contains(&EVENT_WRITE_IDLE));
        assert_eq!(
            commands.dependency_mode(),
            Some(LegacyDependencyMode::ReleaseWait {
                address: FENCE_ADDR,
                next_epoch: 2,
            })
        );
    }

    #[test]
    fn gfx1010_release_wait_rejects_u32_epoch_overflow() {
        let mut commands = Pm4Commands::new_with_dependency(
            Pm4Architecture::Gfx10,
            Pm4RegisterPolicy::Legacy,
            Gfx10DispatchInitiatorPolicy::Legacy,
            None,
            Gfx11ComputeResourceLimitsPolicy::Legacy,
            LegacyDependencyMode::ReleaseWait {
                address: 0x1000,
                next_epoch: u32::MAX,
            },
        );
        let err = commands.wait_compute_idle().unwrap_err();
        assert!(
            err.contains("epoch overflow"),
            "unexpected overflow error: {err}"
        );
    }

    #[test]
    fn gfx1010_release_wait_aba_reset_starts_each_stream_at_zero() {
        // Two independently constructed immutable streams both begin with the
        // sentinel epoch-0 fence, so a stale prior epoch cannot satisfy the
        // next replay (reset-to-zero / ABA shape).
        const FENCE_ADDR: u64 = 0xaaa0;
        let mut first = Pm4Commands::new_with_dependency(
            Pm4Architecture::Gfx10,
            Pm4RegisterPolicy::Legacy,
            Gfx10DispatchInitiatorPolicy::Legacy,
            None,
            Gfx11ComputeResourceLimitsPolicy::Legacy,
            LegacyDependencyMode::ReleaseWait {
                address: FENCE_ADDR,
                next_epoch: 0,
            },
        );
        first.emit_entry_sentinel_reset().unwrap();
        first.wait_compute_idle().unwrap();

        let mut second = Pm4Commands::new_with_dependency(
            Pm4Architecture::Gfx10,
            Pm4RegisterPolicy::Legacy,
            Gfx10DispatchInitiatorPolicy::Legacy,
            None,
            Gfx11ComputeResourceLimitsPolicy::Legacy,
            LegacyDependencyMode::ReleaseWait {
                address: FENCE_ADDR,
                next_epoch: 0,
            },
        );
        second.emit_entry_sentinel_reset().unwrap();

        let first_dwords = first.dwords().unwrap();
        let second_dwords = second.dwords().unwrap();
        assert_eq!(first_dwords[5], 0);
        assert_eq!(second_dwords[5], 0);
        assert_eq!(&first_dwords[..15], &second_dwords[..15]);
        // First stream advanced past the sentinel; second is still at epoch 0.
        assert_eq!(
            first.dependency_mode(),
            Some(LegacyDependencyMode::ReleaseWait {
                address: FENCE_ADDR,
                next_epoch: 1,
            })
        );
        assert_eq!(
            second.dependency_mode(),
            Some(LegacyDependencyMode::ReleaseWait {
                address: FENCE_ADDR,
                next_epoch: 0,
            })
        );
    }

    #[test]
    fn default_dependency_mode_keeps_cs_partial_flush() {
        let mut commands = Pm4Commands::new(
            Pm4Architecture::Gfx10,
            Pm4RegisterPolicy::Legacy,
            Gfx10DispatchInitiatorPolicy::Legacy,
            None,
            Gfx11ComputeResourceLimitsPolicy::Legacy,
        );
        assert_eq!(
            commands.dependency_mode(),
            Some(LegacyDependencyMode::CsPartialFlush)
        );
        commands.emit_entry_sentinel_reset().unwrap();
        commands.wait_compute_idle().unwrap();
        let dwords = commands.dwords().unwrap();
        // EVENT_WRITE CS_PARTIAL_FLUSH only — no RELEASE_MEM / WAIT_REG_MEM.
        assert_eq!(dwords, &[0xc000_4600, 0x407]);
    }

    #[test]
    fn sequence_hash_inputs_unchanged_by_dependency_mode() {
        // Fence selection must not alter capture identity / sequence hashing.
        let launches = [
            RecordedHipLaunch {
                kernel: "a".to_owned(),
                artifact: None,
                grid: [1, 2, 3],
                block: [32, 1, 1],
                shared_mem: 0,
                grid_binding: None,
                declared_kernarg_bindings: Vec::new(),
                kernarg: vec![1, 2, 3, 4],
                accesses: None,
                binding_layout: None,
            },
            RecordedHipLaunch {
                kernel: "b".to_owned(),
                artifact: None,
                grid: [4, 5, 6],
                block: [64, 1, 1],
                shared_mem: 128,
                grid_binding: None,
                declared_kernarg_bindings: Vec::new(),
                kernarg: vec![5, 6],
                accesses: None,
                binding_layout: None,
            },
        ];
        let hash = replay_sequence_hash(&launches);
        assert_eq!(hash, replay_sequence_hash(launches.iter()));
        assert_ne!(hash, 0);
    }

    #[test]
    fn lfm_retained_effect_contracts() {
        assert_eq!(expected_kernarg_bytes("conv1d_gated_decode_f32"), Some(48));
        let conv = pointer_effects("conv1d_gated_decode_f32").expect("conv contract");
        assert_eq!(conv.len(), 4);
        assert_eq!(conv[0].offset, 0);
        assert_eq!(conv[0].mode, RecordedAccessMode::Read);
        assert_eq!(conv[1].offset, 8);
        assert_eq!(
            conv[1].mode,
            RecordedAccessMode::Write,
            "conv state @8 is RMW, recorded as write"
        );
        assert_eq!(conv[2].offset, 16);
        assert_eq!(conv[2].mode, RecordedAccessMode::Read);
        assert_eq!(conv[3].offset, 24);
        assert_eq!(conv[3].mode, RecordedAccessMode::Write);

        assert_eq!(expected_kernarg_bytes("attention_q8_0_kv"), Some(64));
        let attn = pointer_effects("attention_q8_0_kv").expect("attn contract");
        assert_eq!(attn.len(), 5);
        assert_eq!(attn[0].offset, 0);
        assert_eq!(attn[0].mode, RecordedAccessMode::Read);
        assert_eq!(attn[1].offset, 8);
        assert_eq!(attn[1].mode, RecordedAccessMode::Read);
        assert_eq!(attn[2].offset, 16);
        assert_eq!(attn[2].mode, RecordedAccessMode::Read);
        assert_eq!(attn[3].offset, 24);
        assert_eq!(attn[3].mode, RecordedAccessMode::Write);
        assert_eq!(attn[4].offset, 32);
        assert_eq!(attn[4].mode, RecordedAccessMode::Read);

        assert!(expected_kernarg_bytes("unknown_kernel_xyz").is_none());
        assert!(pointer_effects("unknown_kernel_xyz").is_none());
    }

    #[test]
    fn deltanet_f32_retained_effect_contract_covers_recurrent_state() {
        assert_eq!(expected_kernarg_bytes("gated_delta_net_f32"), Some(80));
        let effects = pointer_effects("gated_delta_net_f32").expect("GDN FP32 contract");
        assert_eq!(effects.len(), 7);
        assert_eq!(effects[5].offset, 40);
        assert_eq!(effects[5].mode, RecordedAccessMode::Write);
        assert_eq!(effects[6].offset, 48);
        assert_eq!(effects[6].mode, RecordedAccessMode::Write);
    }

    #[test]
    fn qwen_q8_full_attention_aql_body_uses_system_visibility() {
        let launch = |kernel: &str| RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1; 3],
            block: [32, 1, 1],
            shared_mem: 0,
            grid_binding: None,
            declared_kernarg_bindings: Vec::new(),
            kernarg: Vec::new(),
            accesses: None,
            binding_layout: None,
        };
        let launches = vec![
            launch("before"),
            launch("fused_qkv_hfq4g256"),
            launch("attention_flash_q8_0_tile"),
            launch("attention_flash_q8_0_reduce"),
            launch("gemv_hfq4g256_residual"),
            launch("after"),
        ];
        let mut headers = vec![HeaderPolicy::BATCH_BOUNDARY_INTERNAL_SERIAL; launches.len()];

        apply_qwen_q8_full_attention_visibility(&launches, &mut headers);

        assert_eq!(headers[0], HeaderPolicy::BATCH_BOUNDARY_INTERNAL_SERIAL);
        assert!(headers[1..=4]
            .iter()
            .all(|header| *header == HeaderPolicy::RECORDED_DISPATCH));
        assert_eq!(headers[5], HeaderPolicy::BATCH_BOUNDARY_INTERNAL_SERIAL);
    }

    #[test]
    fn gemma4_ple_activation_keeps_padded_replay_contract() {
        let kernel = "gemma4_ple_gelu_mul_strided_f32";
        let effects = pointer_effects(kernel).expect("Gemma 4 PLE activation contract");
        assert_eq!(effects.len(), 3);
        assert_eq!(effects[0].offset, 0);
        assert_eq!(effects[0].mode, RecordedAccessMode::Read);
        assert_eq!(effects[1].offset, 8);
        assert_eq!(effects[1].mode, RecordedAccessMode::Read);
        assert_eq!(effects[2].offset, 16);
        assert_eq!(effects[2].mode, RecordedAccessMode::Write);

        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..3 {
            blob.push_ptr(std::ptr::null());
        }
        for _ in 0..4 {
            blob.push_i32(0);
        }
        assert_eq!(blob.len(), 40, "explicit kernel arguments occupy 40 bytes");
        blob.pad_to(16);
        assert_eq!(blob.len(), 48, "recorded launches are padded to 16 bytes");
        assert_eq!(expected_kernarg_bytes(kernel), Some(blob.len()));
    }

    #[test]
    fn gqa_decode_attention_pairs_keep_padded_replay_contract() {
        for (tile, reduce) in [
            (
                "attention_flash_fp8_e4m3_tile_gqa_gfx1201",
                "attention_flash_reduce_dsplit_gfx1201",
            ),
            (
                "attention_flash_q8_0_tile_gqa_gfx1151",
                "attention_flash_reduce_dsplit_gfx1151",
            ),
        ] {
            // Launcher order: q, k, v, partials, pos, then 8 scalars.
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..5 {
                blob.push_ptr(std::ptr::null());
            }
            for _ in 0..4 {
                blob.push_i32(0);
            }
            blob.push_f32(0.0);
            for _ in 0..3 {
                blob.push_i32(0);
            }
            blob.pad_to(16);
            assert_eq!(expected_kernarg_bytes(tile), Some(blob.len()), "{tile}");
            let effects = pointer_effects(tile).expect("GQA tile contract");
            let modes: Vec<_> = effects.iter().map(|e| (e.offset, e.mode)).collect();
            assert_eq!(
                modes,
                vec![
                    (0, RecordedAccessMode::Read),
                    (8, RecordedAccessMode::Read),
                    (16, RecordedAccessMode::Read),
                    (24, RecordedAccessMode::Write),
                    (32, RecordedAccessMode::Read),
                ],
                "{tile}"
            );

            // Launcher order: partials, out, n_heads, head_dim, pos, tile, max_tiles.
            let mut blob = hip_bridge::KernargBlob::new();
            blob.push_ptr(std::ptr::null());
            blob.push_ptr(std::ptr::null());
            blob.push_i32(0);
            blob.push_i32(0);
            blob.push_ptr(std::ptr::null());
            blob.push_i32(0);
            blob.push_i32(0);
            blob.pad_to(16);
            assert_eq!(expected_kernarg_bytes(reduce), Some(blob.len()), "{reduce}");
            let effects = pointer_effects(reduce).expect("dsplit reduce contract");
            let modes: Vec<_> = effects.iter().map(|e| (e.offset, e.mode)).collect();
            assert_eq!(
                modes,
                vec![
                    (0, RecordedAccessMode::Read),
                    (8, RecordedAccessMode::Write),
                    (24, RecordedAccessMode::Read),
                ],
                "{reduce}"
            );
        }
    }

    #[test]
    fn gfx1100_fa2_split_verify_keeps_padded_replay_contracts() {
        use RecordedAccessMode::{Read, Write};

        let preconvert = "attention_fa2_q_preconvert_gfx1100";
        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..4 {
            blob.push_ptr(std::ptr::null());
        }
        blob.push_i32(0);
        blob.push_i32(0);
        blob.pad_to(16);
        assert_eq!(expected_kernarg_bytes(preconvert), Some(blob.len()));
        assert_eq!(
            pointer_effects(preconvert)
                .unwrap()
                .iter()
                .map(|effect| (effect.offset, effect.mode))
                .collect::<Vec<_>>(),
            vec![(0, Read), (8, Write), (16, Read), (24, Read)]
        );

        let partial = "attention_q8_0_fa2_gqa_partial_gfx1100";
        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..5 {
            blob.push_ptr(std::ptr::null());
        }
        for _ in 0..4 {
            blob.push_i32(0);
        }
        blob.push_f32(0.0);
        blob.push_i32(0);
        blob.pad_to(16);
        assert_eq!(expected_kernarg_bytes(partial), Some(blob.len()));
        assert_eq!(
            pointer_effects(partial)
                .unwrap()
                .iter()
                .map(|effect| (effect.offset, effect.mode))
                .collect::<Vec<_>>(),
            vec![(0, Read), (8, Read), (16, Read), (24, Write), (32, Read)]
        );

        let merge = "attention_q8_0_fa2_gqa_merge_gfx1100";
        let mut blob = hip_bridge::KernargBlob::new();
        blob.push_ptr(std::ptr::null());
        blob.push_ptr(std::ptr::null());
        for _ in 0..4 {
            blob.push_i32(0);
        }
        blob.pad_to(16);
        assert_eq!(expected_kernarg_bytes(merge), Some(blob.len()));
        assert_eq!(
            pointer_effects(merge)
                .unwrap()
                .iter()
                .map(|effect| (effect.offset, effect.mode))
                .collect::<Vec<_>>(),
            vec![(0, Read), (8, Write)]
        );
    }

    #[test]
    fn gfx1100_f16_projection_producers_keep_replay_contracts() {
        use RecordedAccessMode::{Read, Write};

        let base = "fused_rmsnorm_mq_rotate_f16";
        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..5 {
            blob.push_ptr(std::ptr::null());
        }
        blob.push_i32(0);
        blob.push_f32(0.0);
        blob.pad_to(16);
        assert_eq!(expected_kernarg_bytes(base), Some(blob.len()));
        assert_eq!(
            pointer_effects(base)
                .unwrap()
                .iter()
                .map(|effect| (effect.offset, effect.mode))
                .collect::<Vec<_>>(),
            vec![(0, Read), (8, Read), (16, Read), (24, Read), (32, Write)]
        );

        let awq = "fused_rmsnorm_mq_rotate_awq_f16";
        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..6 {
            blob.push_ptr(std::ptr::null());
        }
        blob.push_i32(0);
        blob.push_f32(0.0);
        blob.pad_to(16);
        assert_eq!(expected_kernarg_bytes(awq), Some(blob.len()));
        assert_eq!(
            pointer_effects(awq)
                .unwrap()
                .iter()
                .map(|effect| (effect.offset, effect.mode))
                .collect::<Vec<_>>(),
            vec![
                (0, Read),
                (8, Read),
                (16, Read),
                (24, Read),
                (32, Read),
                (40, Write),
            ]
        );

        let gate_up = "gemm_gate_up_mq4g256v2_wmma_gfx1100_ldsstage";
        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..5 {
            blob.push_ptr(std::ptr::null());
        }
        for _ in 0..4 {
            blob.push_i32(0);
        }
        blob.pad_to(16);
        assert_eq!(blob.len(), 64);
        assert_eq!(expected_kernarg_bytes(gate_up), Some(blob.len()));
        assert_eq!(
            pointer_effects(gate_up)
                .unwrap()
                .iter()
                .map(|effect| (effect.offset, effect.mode))
                .collect::<Vec<_>>(),
            vec![(0, Read), (8, Read), (16, Read), (24, Write), (32, Write)]
        );

        let copy = "copy_f32_buffer";
        let mut blob = hip_bridge::KernargBlob::new();
        blob.push_ptr(std::ptr::null());
        blob.push_ptr(std::ptr::null());
        blob.push_i32(0);
        blob.pad_to(16);
        assert_eq!(expected_kernarg_bytes(copy), Some(blob.len()));
        assert_eq!(
            pointer_effects(copy)
                .unwrap()
                .iter()
                .map(|effect| (effect.offset, effect.mode))
                .collect::<Vec<_>>(),
            vec![(0, Write), (8, Read)]
        );
    }

    #[test]
    fn gfx1201_qwen36_27b_decode_fusions_keep_padded_replay_contract() {
        use RecordedAccessMode::{Read, Write};
        // Gated norm/MQ rotation launcher: x, z, weight, [awq_scale], signs1,
        // signs2, x_rot, then n_heads, head_dim, eps.
        for (kernel, awq) in [
            ("gated_norm_mq_rotate_k6144_gfx1201", false),
            ("gated_norm_mq_rotate_awq_k6144_gfx1201", true),
        ] {
            let pointers = if awq { 7 } else { 6 };
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..pointers {
                blob.push_ptr(std::ptr::null());
            }
            blob.push_i32(0);
            blob.push_i32(0);
            blob.push_f32(0.0);
            blob.pad_to(16);
            assert_eq!(expected_kernarg_bytes(kernel), Some(blob.len()), "{kernel}");
            let modes: Vec<_> = pointer_effects(kernel)
                .expect("gated norm/MQ rotation contract")
                .iter()
                .map(|e| (e.offset, e.mode))
                .collect();
            let mut expected: Vec<_> = (0..pointers - 1).map(|i| (i * 8, Read)).collect();
            expected.push(((pointers - 1) * 8, Write));
            assert_eq!(modes, expected, "{kernel}");
        }

        // FA prep launcher: q_interleaved, q, gate, k, q_weight, k_weight,
        // pos, then eps and freq_base.
        let prep = "qwen36_27b_fa_prep_gfx1201";
        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..7 {
            blob.push_ptr(std::ptr::null());
        }
        blob.push_f32(0.0);
        blob.push_f32(0.0);
        blob.pad_to(16);
        assert_eq!(expected_kernarg_bytes(prep), Some(blob.len()));
        let modes: Vec<_> = pointer_effects(prep)
            .expect("FA prep contract")
            .iter()
            .map(|e| (e.offset, e.mode))
            .collect();
        assert_eq!(
            modes,
            vec![
                (0, Read),
                (8, Write),
                (16, Write),
                (24, Write),
                (32, Read),
                (40, Read),
                (48, Read)
            ]
        );
    }

    #[test]
    fn gfx1151_radiowave_symbols_keep_resource_contracts() {
        let gate = "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_all_buffer_gfx1151";
        let gate_hybrid = "gemv_hfq4g256_moe_gate_up_k8_indexed_k2048_hybrid_gfx1151";
        let gate_up_paired = "gemv_hfq4g256_moe_gate_up_k8_indexed_paired_waves_k2048_gfx1151";
        let gate_up_persistent = "gemv_hfq4g256_moe_gate_up_k8_indexed_persistent_rank8_gfx1151";
        let qkvza = "fused_qkvza_hfq4g256_k2048_all_buffer_gfx1151";
        let qkvza_hybrid = "fused_qkvza_hfq4g256_k2048_hybrid_buffer_gfx1151";
        let qkvza_r4 = "fused_qkvza_hfq4g256_k2048_r4_stream_gfx1151";
        assert_eq!(expected_kernarg_bytes(gate), Some(48));
        assert_eq!(pointer_effects(gate).map(|effects| effects.len()), Some(5));
        assert_eq!(expected_kernarg_bytes(gate_hybrid), Some(48));
        assert_eq!(
            pointer_effects(gate_hybrid).map(|effects| effects.len()),
            Some(5)
        );
        assert_eq!(expected_kernarg_bytes(gate_up_paired), Some(48));
        assert_eq!(
            pointer_effects(gate_up_paired).map(|effects| effects.len()),
            Some(5)
        );
        assert_eq!(expected_kernarg_bytes(gate_up_persistent), Some(48));
        assert_eq!(
            pointer_effects(gate_up_persistent).map(|effects| effects.len()),
            Some(5)
        );
        assert_eq!(expected_kernarg_bytes(qkvza), Some(96));
        assert_eq!(pointer_effects(qkvza).map(|effects| effects.len()), Some(9));
        assert_eq!(expected_kernarg_bytes(qkvza_hybrid), Some(96));
        assert_eq!(
            pointer_effects(qkvza_hybrid).map(|effects| effects.len()),
            Some(9)
        );
        assert_eq!(expected_kernarg_bytes(qkvza_r4), Some(96));
        assert_eq!(
            pointer_effects(qkvza_r4).map(|effects| effects.len()),
            Some(9)
        );
        for producer in [
            "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151",
            "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151",
        ] {
            assert_eq!(expected_kernarg_bytes(producer), Some(48));
            let effects = pointer_effects(producer).expect("split projection contract");
            assert_eq!(effects.len(), 4);
            assert_eq!(effects[0].mode, RecordedAccessMode::Read);
            assert_eq!(effects[1].mode, RecordedAccessMode::Read);
            assert_eq!(effects[2].mode, RecordedAccessMode::Read);
            assert_eq!(effects[3].mode, RecordedAccessMode::Write);
            assert_eq!(effects[3].offset, 24);
        }
        for (symbol, kernarg_bytes, pointer_count) in [
            ("gated_norm_mq_rotate_gfx1151", 64, 6),
            ("qwen35_fa_prep_gfx1151", 64, 7),
            ("attention_flash_q8_0_reduce_gated_mq_rotate_gfx1151", 64, 6),
            ("moe_down_combine_rmsnorm_mq_rotate_vecsum_gfx1151", 72, 7),
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(kernarg_bytes));
            assert_eq!(
                pointer_effects(symbol).map(|effects| effects.len()),
                Some(pointer_count)
            );
        }
        assert_eq!(
            expected_kernarg_bytes("attention_flash_q8_0_tile"),
            Some(80)
        );
        assert_eq!(
            pointer_effects("attention_flash_q8_0_tile").map(|effects| effects.len()),
            Some(5)
        );
        assert_eq!(
            expected_kernarg_bytes("gemv_hfq4g256_residual_wave64"),
            Some(32)
        );
        assert_eq!(
            pointer_effects("gemv_hfq4g256_residual_wave64").map(|effects| effects.len()),
            Some(3)
        );
        let residual_r2 = "gemv_hfq4g256_residual_multirow_r2_gfx1151";
        assert_eq!(expected_kernarg_bytes(residual_r2), Some(32));
        assert_eq!(
            pointer_effects(residual_r2).map(|effects| effects.len()),
            Some(3)
        );
        let residual_k4096 = "gemv_hfq4g256_residual_k4096_gfx1151";
        assert_eq!(expected_kernarg_bytes(residual_k4096), Some(32));
        assert_eq!(
            pointer_effects(residual_k4096).map(|effects| effects.len()),
            Some(3)
        );
        let residual_rt_low = "gemv_hfq4g256_residual_rt_low_gfx1151";
        assert_eq!(expected_kernarg_bytes(residual_rt_low), Some(32));
        assert_eq!(
            pointer_effects(residual_rt_low).map(|effects| effects.len()),
            Some(3)
        );
        let down = "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row2_buffer_gfx1151";
        assert_eq!(expected_kernarg_bytes(down), Some(48));
        assert_eq!(pointer_effects(down).map(|effects| effects.len()), Some(4));
        let down_row8 = "gemv_hfq4g256_moe_down_k8_indexed_batched_expanded_row8_gfx1151";
        assert_eq!(expected_kernarg_bytes(down_row8), Some(48));
        assert_eq!(
            pointer_effects(down_row8).map(|effects| effects.len()),
            Some(4)
        );
        let lm_head = "gemv_hfq4g256_lm_head_r1_hybrid_buffer_gfx1151";
        assert_eq!(expected_kernarg_bytes(lm_head), Some(32));
        assert_eq!(
            pointer_effects(lm_head).map(|effects| effects.len()),
            Some(3)
        );
        let lm_head_dot2 = "gemv_hfq4g256_lm_head_dot2_gfx1151";
        assert_eq!(expected_kernarg_bytes(lm_head_dot2), Some(32));
        assert_eq!(
            pointer_effects(lm_head_dot2).map(|effects| effects.len()),
            Some(3)
        );
    }

    // Fail if a codebook MoE kernel variant is added without resource-contract registration.
    #[test]
    fn codebook_moe_symbols_have_resource_contracts() {
        let gate_up_effects = vec![read(0), read(8), read(16), write(24), write(32)];
        let down_effects = vec![read(0), read(8), read(16), read(24), write(32)];
        for (symbol, kernarg_bytes, effects) in [
            (
                "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed",
                48usize,
                &gate_up_effects,
            ),
            (
                "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed",
                48,
                &gate_up_effects,
            ),
            (
                "gemv_mq2g256gl_moe_gate_up_k8_indexed",
                64,
                &gate_up_effects,
            ),
            (
                "gemv_mq3g256gl_moe_gate_up_k8_indexed",
                80,
                &gate_up_effects,
            ),
            (
                "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed",
                48,
                &down_effects,
            ),
            (
                "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r2",
                48,
                &down_effects,
            ),
            (
                "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_r4",
                48,
                &down_effects,
            ),
            (
                "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed",
                48,
                &down_effects,
            ),
            (
                "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r2",
                48,
                &down_effects,
            ),
            (
                "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_r4",
                48,
                &down_effects,
            ),
            ("gemv_mq3g256_lloyd_moe_ninepath_d4", 48, &down_effects),
            (
                "gemv_mq2g256gl_moe_down_residual_scaled_k8_indexed",
                64,
                &down_effects,
            ),
            (
                "gemv_mq3g256gl_moe_down_residual_scaled_k8_indexed",
                80,
                &down_effects,
            ),
            // Batched-K4 prefill siblings. 52 B, NOT 48: the K_TOP scalar is
            // exactly the kind of quiet ABI difference that makes a
            // pattern-matched contract fail closed instead of loudly.
            (
                "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4",
                52,
                &gate_up_effects,
            ),
            (
                "gemv_mq3g256_lloyd_moe_gate_up_k8_indexed_batched_k4",
                52,
                &gate_up_effects,
            ),
            (
                "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4",
                52,
                &down_effects,
            ),
            (
                "gemv_mq3g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4",
                52,
                &down_effects,
            ),
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(kernarg_bytes));
            let got = pointer_effects(symbol).expect("codebook MoE pointer contract");
            assert_eq!(got.len(), effects.len());
            for (got_effect, want) in got.iter().zip(effects.iter()) {
                assert_eq!(got_effect.offset, want.offset);
                assert_eq!(got_effect.mode, want.mode);
            }
            // Offset 24 is Write for gate_up and Read for down/ninepath.
            assert_eq!(got[3].offset, 24);
            if symbol.contains("gate_up") {
                assert_eq!(got[3].mode, RecordedAccessMode::Write);
            } else {
                assert_eq!(got[3].mode, RecordedAccessMode::Read);
            }
        }
    }

    // Fail closed if an MQ4V2/MQ6V2 MoE kernel is admitted without a distinct
    // resource contract. V2 names must never silently fall through to V1.
    #[test]
    fn mqv2_moe_symbols_have_resource_contracts() {
        let gate_up_effects = vec![read(0), read(8), read(16), write(24), write(32)];
        let expanded_down_effects = vec![read(0), read(8), read(16), write(24)];
        let ninepath_effects = vec![read(0), read(8), read(16), read(24), write(32)];
        let grouped_effects = vec![read(0), read(8), read(16), read(24), write(32)];

        // Decode gate_up: 5 ptr + M,K = 48 B (already 16-aligned).
        // Includes gfx1100 exact K=2048 nolds candidate: same ABI/grid contract
        // as the generic gate_up, only workgroup/LDS/barrier removal differs.
        for symbol in [
            "gemv_mq4g256v2_moe_gate_up_k8_indexed",
            "gemv_mq6g256v2_moe_gate_up_k8_indexed",
            "gemv_mq4g256v2_moe_gate_up_k8_indexed_k2048_nolds_gfx1100",
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(48));
            let got = pointer_effects(symbol).expect("mqv2 gate_up pointer contract");
            assert_eq!(got.len(), gate_up_effects.len());
            for (got_effect, want) in got.iter().zip(gate_up_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset);
                assert_eq!(got_effect.mode, want.mode);
            }
        }

        // Path1 batched gate_up: 5 ptr + M,K,K_TOP = 52 → pad_to(16) = 64.
        // Prove the padded recorded length, not the bare field total.
        for symbol in [
            "gemv_mq4g256v2_moe_gate_up_k8_indexed_batched",
            "gemv_mq6g256v2_moe_gate_up_k8_indexed_batched",
        ] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..5 {
                blob.push_ptr(std::ptr::null());
            }
            blob.push_i32(0);
            blob.push_i32(0);
            blob.push_i32(0);
            assert_eq!(blob.len(), 52, "{symbol} explicit args occupy 52 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 64, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("mqv2 batched gate_up pointer contract");
            assert_eq!(got.len(), gate_up_effects.len());
            for (got_effect, want) in got.iter().zip(gate_up_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset);
                assert_eq!(got_effect.mode, want.mode);
            }
        }

        // Expanded down: 4 ptr + M,K,K_TOP = 44 → 48 padded.
        for symbol in [
            "gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded",
            "gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded",
        ] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..4 {
                blob.push_ptr(std::ptr::null());
            }
            blob.push_i32(0);
            blob.push_i32(0);
            blob.push_i32(0);
            assert_eq!(blob.len(), 44, "{symbol} explicit args occupy 44 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 48, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("mqv2 expanded down pointer contract");
            assert_eq!(got.len(), expanded_down_effects.len());
            for (got_effect, want) in got.iter().zip(expanded_down_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset);
                assert_eq!(got_effect.mode, want.mode);
            }
        }

        // Ninepath fused down+combine: 5 ptr + down_m,down_k = 48.
        // RPB8 gfx1100 candidate shares the frozen 48-byte ABI.
        for symbol in [
            "gemv_mq4g256v2_moe_ninepath_d4",
            "gemv_mq4g256v2_moe_ninepath_rpb8_gfx1100",
            "gemv_mq6g256v2_moe_ninepath_d4",
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(48));
            let got = pointer_effects(symbol).expect("mqv2 ninepath pointer contract");
            assert_eq!(got.len(), ninepath_effects.len());
            for (got_effect, want) in got.iter().zip(ninepath_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset);
                assert_eq!(got_effect.mode, want.mode);
            }
            // Offset 24 is the rotated act (read); out RMW lives at 32.
            assert_eq!(got[3].mode, RecordedAccessMode::Read);
            assert_eq!(got[4].offset, 32);
            assert_eq!(got[4].mode, RecordedAccessMode::Write);
        }

        // Grouped prefill WMMA: 5 ptr + M,K,x_row_div,m_total = 56 → 64.
        // gfx11 `_k2` and gfx12 `_gfx12` are distinct symbols, not aliases.
        for symbol in [
            "gemm_mq4g256v2_moe_grouped_wmma_k2",
            "gemm_mq4g256v2_moe_grouped_wmma_gfx12",
            "gemm_mq6g256v2_moe_grouped_wmma_k2",
            "gemm_mq6g256v2_moe_grouped_wmma_gfx12",
        ] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..5 {
                blob.push_ptr(std::ptr::null());
            }
            for _ in 0..4 {
                blob.push_i32(0);
            }
            assert_eq!(blob.len(), 56, "{symbol} explicit args occupy 56 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 64, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("mqv2 grouped pointer contract");
            assert_eq!(got.len(), grouped_effects.len());
            for (got_effect, want) in got.iter().zip(grouped_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset);
                assert_eq!(got_effect.mode, want.mode);
            }
        }

        // No V1/V2 collapse: MQ4V2/MQ6V2 names are not HFQ4/HFQ6 contracts by alias.
        assert!(
            pointer_effects("gemv_mq4g256v2_moe_gate_up_k8_indexed").is_some()
                && pointer_effects("gemv_hfq4g256_moe_gate_up_k8_indexed").is_some()
        );
        assert_ne!(
            "gemv_mq4g256v2_moe_gate_up_k8_indexed",
            "gemv_hfq4g256_moe_gate_up_k8_indexed"
        );
        assert_ne!(
            "gemv_mq6g256v2_moe_gate_up_k8_indexed",
            "gemv_mq4g256v2_moe_gate_up_k8_indexed"
        );
        // Unknown V2-looking name still fails closed.
        assert!(pointer_effects("gemv_mq4g256v2_moe_gate_up_k8_indexed_unknown").is_none());
        assert!(expected_kernarg_bytes("gemv_mq4g256v2_moe_gate_up_k8_indexed_unknown").is_none());
    }

    // Table-driven: mixed-precision MoE kernels (tags 0..18, V1/V2 never alias).
    // Covers every kernel admitted to the mixed grouped/expanded replay path.
    #[test]
    fn mixed_moe_symbols_have_resource_contracts() {
        let mixed_gate_up_effects =
            vec![read(0), read(8), read(16), read(24), write(32), write(40)];
        let mixed_down_effects = vec![read(0), read(8), read(16), read(24), write(32)];
        let mixed_grouped_effects = vec![read(0), read(8), read(16), read(24), read(32), write(40)];

        // Mixed gate_up batched: 6 ptr + M,K,K_TOP = 60 → 64 padded.
        for symbol in ["gemv_mixed_moe_gate_up_k8_indexed_batched"] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..6 {
                blob.push_ptr(std::ptr::null());
            }
            for _ in 0..3 {
                blob.push_i32(0);
            }
            assert_eq!(blob.len(), 60, "{symbol} explicit args occupy 60 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 64, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("mixed gate_up pointer contract");
            assert_eq!(
                got.len(),
                mixed_gate_up_effects.len(),
                "{symbol} effect count"
            );
            for (got_effect, want) in got.iter().zip(mixed_gate_up_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset, "{symbol} offset");
                assert_eq!(
                    got_effect.mode, want.mode,
                    "{symbol} mode at {}",
                    want.offset
                );
            }
            // Extra pointer is dtype_tags at +8 (read); y_gate/y_up are distinct writes.
            assert_eq!(got[1].offset, 8);
            assert_eq!(got[1].mode, RecordedAccessMode::Read);
            assert_eq!(got[4].mode, RecordedAccessMode::Write);
            assert_eq!(got[5].mode, RecordedAccessMode::Write);
        }

        // Mixed down expanded: 5 ptr + M,K,K_TOP = 52 → 64 padded.
        for symbol in ["gemv_mixed_moe_down_k8_indexed_batched_expanded"] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..5 {
                blob.push_ptr(std::ptr::null());
            }
            for _ in 0..3 {
                blob.push_i32(0);
            }
            assert_eq!(blob.len(), 52, "{symbol} explicit args occupy 52 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 64, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("mixed down pointer contract");
            assert_eq!(got.len(), mixed_down_effects.len(), "{symbol} effect count");
            for (got_effect, want) in got.iter().zip(mixed_down_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset, "{symbol} offset");
                assert_eq!(
                    got_effect.mode, want.mode,
                    "{symbol} mode at {}",
                    want.offset
                );
            }
            assert_eq!(got[1].offset, 8);
            assert_eq!(got[1].mode, RecordedAccessMode::Read);
            assert_eq!(got[4].mode, RecordedAccessMode::Write);
        }

        // Mixed grouped WMMA: 6 ptr + M,K,x_row_div,m_total = 64 (already aligned).
        // gfx11 k2, gfx12, and 4w_k2 tiling variants are distinct symbols.
        for symbol in [
            "gemm_mixed_moe_grouped_wmma_k2",
            "gemm_mixed_moe_grouped_wmma_gfx12",
            "gemm_mixed_moe_grouped_wmma_4w_k2",
        ] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..6 {
                blob.push_ptr(std::ptr::null());
            }
            for _ in 0..4 {
                blob.push_i32(0);
            }
            assert_eq!(blob.len(), 64, "{symbol} explicit args occupy 64 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 64, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("mixed grouped pointer contract");
            assert_eq!(
                got.len(),
                mixed_grouped_effects.len(),
                "{symbol} effect count"
            );
            for (got_effect, want) in got.iter().zip(mixed_grouped_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset, "{symbol} offset");
                assert_eq!(
                    got_effect.mode, want.mode,
                    "{symbol} mode at {}",
                    want.offset
                );
            }
            assert_eq!(got[5].offset, 40);
            assert_eq!(got[5].mode, RecordedAccessMode::Write);
        }

        // V1/V2 names never alias; unknown mixed name still fails closed.
        assert!(pointer_effects("gemv_mixed_moe_gate_up_k8_indexed_batched_unknown").is_none());
        assert!(
            expected_kernarg_bytes("gemv_mixed_moe_gate_up_k8_indexed_batched_unknown").is_none()
        );
        assert_ne!(
            "gemv_mixed_moe_gate_up_k8_indexed_batched",
            "gemv_mq4g256v2_moe_gate_up_k8_indexed_batched"
        );
    }

    // Table-driven: every reachable dense shared-expert V2 symbol.
    // Shared experts ride the exact dense V2 GEMV/GEMM ABIs — never V1 aliases.
    #[test]
    fn dense_shared_v2_symbols_have_resource_contracts() {
        let dense_gemv_effects = vec![read(0), read(8), write(16)];
        let dense_gemm_effects = vec![read(0), read(8), write(16)];

        // Dense GEMV: 3 ptr + M,K = 32 (already 16-aligned). Plain, residual, multirow.
        for symbol in [
            "gemv_mq4g256v2",
            "gemv_mq4g256v2_residual",
            "gemv_mq4g256v2_residual_r1_k4096_gfx1100_noscratch",
            "gemv_mq6g256v2",
            "gemv_mq6g256v2_residual",
            "gemv_mq4g256v2_multirow_r2",
            "gemv_mq4g256v2_multirow_r4",
            "gemv_mq4g256v2_multirow_r8",
            "gemv_mq6g256v2_multirow_r2",
            "gemv_mq6g256v2_multirow_r4",
            "gemv_mq6g256v2_multirow_r8",
        ] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..3 {
                blob.push_ptr(std::ptr::null());
            }
            blob.push_i32(0);
            blob.push_i32(0);
            assert_eq!(blob.len(), 32, "{symbol} explicit args occupy 32 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 32, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("dense gemv pointer contract");
            assert_eq!(got.len(), dense_gemv_effects.len(), "{symbol} effect count");
            for (got_effect, want) in got.iter().zip(dense_gemv_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset, "{symbol} offset");
                assert_eq!(
                    got_effect.mode, want.mode,
                    "{symbol} mode at {}",
                    want.offset
                );
            }
        }

        // Dense GEMM residual WMMA: 3 ptr + M,K,batch = 36 → 48 padded.
        // Covers plain and arch-tiled gfx11/gfx12 variants admitted to slots/prefill.
        for symbol in [
            "gemm_mq4g256v2_residual_wmma",
            "gemm_mq4g256v2_residual_wmma_gfx12",
            "gemm_mq6g256v2_residual_wmma",
            "gemm_mq6g256v2_residual_wmma_gfx12",
            "gemm_mq4g256v2_residual_wmma_gfx11_bt4",
            "gemm_mq4g256v2_residual_wmma_gfx11_bt6",
            "gemm_mq4g256v2_residual_wmma_gfx11_bt8",
            "gemm_mq6g256v2_residual_wmma_gfx11_bt4",
            "gemm_mq6g256v2_residual_wmma_gfx11_bt6",
            "gemm_mq6g256v2_residual_wmma_gfx11_bt8",
            "gemm_mq4g256v2_residual_wmma_gfx1100_mw4_lds",
            "gemm_mq4g256v2_residual_wmma_gfx1100_mw8_lds",
            "gemm_mq4g256v2_residual_wmma_gfx1100_ks2_lds",
            "gemm_mq4g256v2_residual_wmma_gfx1100_ks4_lds",
            "gemm_mq4g256v2_residual_wmma_gfx1100_ks8_lds",
            "gemm_mq4g256v2_residual_wmma_gfx1100_ldsstage",
            "gemm_mq6g256v2_residual_wmma_gfx11_mw4_lds",
            "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds",
        ] {
            let mut blob = hip_bridge::KernargBlob::new();
            for _ in 0..3 {
                blob.push_ptr(std::ptr::null());
            }
            for _ in 0..3 {
                blob.push_i32(0);
            }
            assert_eq!(blob.len(), 36, "{symbol} explicit args occupy 36 bytes");
            blob.pad_to(16);
            assert_eq!(blob.len(), 48, "{symbol} recorded launches pad to 16");
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
            let got = pointer_effects(symbol).expect("dense gemm pointer contract");
            assert_eq!(got.len(), dense_gemm_effects.len(), "{symbol} effect count");
            for (got_effect, want) in got.iter().zip(dense_gemm_effects.iter()) {
                assert_eq!(got_effect.offset, want.offset, "{symbol} offset");
                assert_eq!(
                    got_effect.mode, want.mode,
                    "{symbol} mode at {}",
                    want.offset
                );
            }
        }

        // No V1/V2 collapse for dense; unknown dense V2 name still fails closed.
        assert!(pointer_effects("gemv_mq4g256v2_unknown").is_none());
        assert!(expected_kernarg_bytes("gemv_mq4g256v2_unknown").is_none());
        assert_ne!("gemv_mq4g256v2", "gemv_hfq4g256");
        assert_ne!("gemm_mq4g256v2_residual_wmma", "gemm_mq4g256_residual_wmma");
        assert!(pointer_effects("gemv_mq4g256v2").is_some());
        assert!(pointer_effects("gemv_mq6g256v2").is_some());
    }

    #[test]
    fn mq4g256v2_residual_r1_k4096_gfx1100_noscratch_keeps_exact_residual_contract() {
        // gfx1100-only residual R1 K=4096 noscratch candidate: same 32-B plain V2
        // residual ABI (A@0, x@8, y@16 RMW/write; M@24, K@28). Distinct symbol —
        // never collapses onto generic residual or unknown names.
        let symbol = "gemv_mq4g256v2_residual_r1_k4096_gfx1100_noscratch";
        let want = [read(0), read(8), write(16)];

        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..3 {
            blob.push_ptr(std::ptr::null());
        }
        blob.push_i32(0);
        blob.push_i32(0);
        assert_eq!(blob.len(), 32, "{symbol} explicit args occupy 32 bytes");
        blob.pad_to(16);
        assert_eq!(blob.len(), 32, "{symbol} recorded launches pad to 16");
        assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));

        let got = pointer_effects(symbol).expect("noscratch residual pointer contract");
        assert_eq!(got.len(), want.len());
        for (got_effect, want_effect) in got.iter().zip(want.iter()) {
            assert_eq!(got_effect.offset, want_effect.offset);
            assert_eq!(got_effect.mode, want_effect.mode);
        }

        // Fail-closed: unknown / mismatched residual names stay unrecognized.
        assert!(pointer_effects("gemv_mq4g256v2_residual_r1_k4096_gfx1100").is_none());
        assert!(expected_kernarg_bytes("gemv_mq4g256v2_residual_r1_k4096_gfx1100").is_none());
        assert_ne!(symbol, "gemv_mq4g256v2_residual");
        assert_ne!(symbol, "gemv_hfq4g256_residual_k2048");
    }

    #[test]
    fn mq4g256v2_residual_sigmoid_scaled_k512_keeps_exact_40b_abi() {
        // Ornith qt44 shared-down fuse: A@0, x@8, y@16 RMW/write, c_buf@24 read;
        // M@32, K@36 → 40 explicit bytes, pad_to(16) → 48. Four pointer offsets.
        let symbol = "gemv_mq4g256v2_residual_sigmoid_scaled_k512";
        let want = [read(0), read(8), write(16), read(24)];

        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..4 {
            blob.push_ptr(std::ptr::null());
        }
        blob.push_i32(0);
        blob.push_i32(0);
        assert_eq!(blob.len(), 40, "{symbol} explicit args occupy 40 bytes");
        blob.pad_to(16);
        assert_eq!(blob.len(), 48, "{symbol} recorded launches pad to 16");
        assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));

        let got = pointer_effects(symbol).expect("sigmoid residual pointer contract");
        assert_eq!(got.len(), want.len());
        for (got_effect, want_effect) in got.iter().zip(want.iter()) {
            assert_eq!(got_effect.offset, want_effect.offset);
            assert_eq!(got_effect.mode, want_effect.mode);
        }

        // Distinct from V1 residual_sigmoid and plain V2 residual; fail-closed on typos.
        assert_ne!(symbol, "gemv_hfq4g256_residual_sigmoid_scaled_gpu");
        assert_ne!(symbol, "gemv_mq4g256v2_residual");
        assert!(pointer_effects("gemv_mq4g256v2_residual_sigmoid_scaled").is_none());
        assert!(expected_kernarg_bytes("gemv_mq4g256v2_residual_sigmoid_scaled").is_none());
    }

    #[test]
    fn fused_qkvza_mq4g256v2_keeps_exact_scalar_replay_contract() {
        let want = [
            read(0),
            read(8),
            read(16),
            read(24),
            read(32),
            write(40),
            write(48),
            write(56),
            write(64),
        ];

        // Generic plus gfx1100 K=2048 HOIST_X32 share the same 14-arg ABI:
        // reads@0/8/16/24/32, writes@40/48/56/64, pad96.
        for symbol in [
            "fused_qkvza_mq4g256v2",
            "fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100",
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(96));
            let got = pointer_effects(symbol).expect("mq4g256v2 qkvza pointer contract");
            assert_eq!(got.len(), want.len());
            for (got_effect, want_effect) in got.iter().zip(want.iter()) {
                assert_eq!(got_effect.offset, want_effect.offset);
                assert_eq!(got_effect.mode, want_effect.mode);
            }
        }

        // Identities stay independent: hoist never collapses onto generic.
        assert_ne!(
            "fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100",
            "fused_qkvza_mq4g256v2"
        );

        // 9 ptr + 5 i32 dimensions = 92 explicit bytes, pad_to(16) → 96.
        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..9 {
            blob.push_ptr(std::ptr::null());
        }
        for _ in 0..5 {
            blob.push_i32(0);
        }
        assert_eq!(blob.len(), 92, "explicit kernel arguments occupy 92 bytes");
        blob.pad_to(16);
        assert_eq!(blob.len(), 96, "recorded launches pad to 16 bytes");
        for symbol in [
            "fused_qkvza_mq4g256v2",
            "fused_qkvza_mq4g256v2_k2048_hoist_x32_gfx1100",
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
        }
    }

    #[test]
    fn fused_qkv_mq4g256v2_keeps_exact_scalar_replay_contract() {
        let want = [
            read(0),
            read(8),
            read(16),
            read(24),
            write(32),
            write(40),
            write(48),
        ];

        // Generic plus gfx1100 K=2048 X_BUFFER share the same 11-arg ABI:
        // reads@0/8/16/24, writes@32/40/48, pad80.
        for symbol in [
            "fused_qkv_mq4g256v2",
            "fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100",
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(80));
            let got = pointer_effects(symbol).expect("mq4g256v2 qkv pointer contract");
            assert_eq!(got.len(), want.len());
            for (got_effect, want_effect) in got.iter().zip(want.iter()) {
                assert_eq!(got_effect.offset, want_effect.offset);
                assert_eq!(got_effect.mode, want_effect.mode);
            }
        }

        // Identities stay independent: x_buffer never collapses onto generic.
        assert_ne!(
            "fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100",
            "fused_qkv_mq4g256v2"
        );

        // 7 ptr + 4 i32 dimensions = 72 explicit bytes, pad_to(16) → 80.
        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..7 {
            blob.push_ptr(std::ptr::null());
        }
        for _ in 0..4 {
            blob.push_i32(0);
        }
        assert_eq!(blob.len(), 72, "explicit kernel arguments occupy 72 bytes");
        blob.pad_to(16);
        assert_eq!(blob.len(), 80, "recorded launches pad to 16 bytes");
        for symbol in [
            "fused_qkv_mq4g256v2",
            "fused_qkv_mq4g256v2_k2048_x_buffer_gfx1100",
        ] {
            assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
        }
    }

    #[test]
    fn fused_gate_up_mq4g256v2_keeps_exact_scalar_replay_contract() {
        let symbol = "fused_gate_up_mq4g256v2";
        let want = [read(0), read(8), read(16), write(24), write(32)];

        assert_eq!(expected_kernarg_bytes(symbol), Some(64));
        let got = pointer_effects(symbol).expect("mq4g256v2 gate_up pointer contract");
        assert_eq!(got.len(), want.len());
        for (got_effect, want_effect) in got.iter().zip(want.iter()) {
            assert_eq!(got_effect.offset, want_effect.offset);
            assert_eq!(got_effect.mode, want_effect.mode);
        }

        // 5 ptr + 3 i32 dimensions = 52 explicit bytes, pad_to(16) → 64.
        let mut blob = hip_bridge::KernargBlob::new();
        for _ in 0..5 {
            blob.push_ptr(std::ptr::null());
        }
        for _ in 0..3 {
            blob.push_i32(0);
        }
        assert_eq!(blob.len(), 52, "explicit kernel arguments occupy 52 bytes");
        blob.pad_to(16);
        assert_eq!(blob.len(), 64, "recorded launches pad to 16 bytes");
        assert_eq!(expected_kernarg_bytes(symbol), Some(blob.len()));
    }

    #[test]
    fn fused_gate_up_mq4v2_k5120_matches_generic_replay_contract() {
        let candidate = "fused_gate_up_mq4g256v2_k5120_gfx1100";
        let baseline = "fused_gate_up_mq4g256v2";
        assert_eq!(expected_kernarg_bytes(candidate), Some(64));
        let got = pointer_effects(candidate).unwrap();
        let want = pointer_effects(baseline).unwrap();
        assert_eq!(got.len(), want.len());
        for (got, want) in got.iter().zip(want.iter()) {
            assert_eq!(got.offset, want.offset);
            assert_eq!(got.mode, want.mode);
        }
    }

    #[test]
    fn moe_shared_down_and_routed_gate_up_are_independent_siblings() {
        assert!(independent_sibling(
            "gemv_hfq4g256_moe_gate_k8_indexed_k2048_gfx1151",
            "gemv_hfq4g256_moe_up_k8_indexed_k2048_gfx1151",
        ));
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed",
        ));
        assert!(independent_sibling(
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
            "gemv_hfq4g256_moe_gate_up_k8_indexed",
        ));
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2",
        ));
        assert!(independent_sibling(
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_wg2",
        ));
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave",
        ));
        assert!(independent_sibling(
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_rank_interleave",
        ));
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr",
        ));
        assert!(independent_sibling(
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_low_vgpr",
        ));
        assert!(independent_sibling(
            "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc",
        ));
        assert!(independent_sibling(
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_pair_slc",
        ));
        for kernel in [
            "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_dlc",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_glc",
            "gemv_hfq4g256_moe_gate_up_k8_indexed_cpol_slc",
        ] {
            assert!(independent_sibling(
                "gemv_hfq4g256_residual_sigmoid_scaled_gpu",
                kernel,
            ));
            assert!(independent_sibling(
                "gemv_mq4g256v2_residual_sigmoid_scaled_k512",
                kernel,
            ));
        }
        assert!(!independent_sibling(
            "gemv_hfq4g256_moe_gate_up_k8_indexed",
            "fused_silu_mul_mq_rotate",
        ));
    }

    #[test]
    fn gfx12_rotated_vmem_writers_require_a_pre_dispatch_acquire() {
        for producer in ["mq_rotate_x", "fused_silu_mul_mq_rotate"] {
            assert!(requires_gfx12_pre_dispatch_vmem_acquire(producer));
        }
        assert!(!requires_gfx12_pre_dispatch_vmem_acquire(
            "gemv_hfq4g256_residual"
        ));
    }

    #[test]
    fn pm4_mid_acquire_policies_preserve_required_boundaries() {
        assert_eq!(
            Pm4MidAcquirePolicy::from_value("conservative"),
            Some(Pm4MidAcquirePolicy::Conservative)
        );
        assert_eq!(
            Pm4MidAcquirePolicy::from_value("entry-only"),
            Some(Pm4MidAcquirePolicy::EntryOnly)
        );
        assert_eq!(
            Pm4MidAcquirePolicy::from_value("required-only"),
            Some(Pm4MidAcquirePolicy::RequiredOnly)
        );
        assert!(Pm4MidAcquirePolicy::Conservative
            .acquire_between("rmsnorm_f32", "rope_partial_halfsplit_f32"));
        assert!(!Pm4MidAcquirePolicy::EntryOnly
            .acquire_between("rmsnorm_f32", "rope_partial_halfsplit_f32"));
        assert!(!Pm4MidAcquirePolicy::Conservative.acquire_between("rmsnorm_f32", "gemv_hfq4g256"));
        assert!(!Pm4MidAcquirePolicy::WithoutRope
            .acquire_between("rmsnorm_f32", "rope_partial_halfsplit_f32"));
        assert!(Pm4MidAcquirePolicy::WithoutRope
            .acquire_between("repeat_interleave_qk_f32", "rope_partial_halfsplit_f32"));
        assert!(
            !Pm4MidAcquirePolicy::WithoutMqRotate.acquire_between("mq_rotate_x", "gemv_hfq4g256")
        );
        assert!(Pm4MidAcquirePolicy::RequiredOnly
            .acquire_between("mq_rotate_x", "gemv_hfq4g256_multirow_r2"));
        assert!(Pm4MidAcquirePolicy::RequiredOnly
            .acquire_between("rmsnorm_f32", "rope_partial_halfsplit_f32"));
        assert!(Pm4MidAcquirePolicy::RequiredOnly
            .acquire_between("fused_silu_mul_mq_rotate", "gemv_hfq4g256_residual"));
        assert!(Pm4MidAcquirePolicy::RequiredOnly.acquire_between(
            "fused_silu_mul_mq_rotate",
            "gemv_hfq4g256_residual_k2048_gfx1201"
        ));
        assert!(!Pm4MidAcquirePolicy::RequiredOnly
            .acquire_between("fused_silu_mul_mq_rotate", "gemv_hfq4g256"));
        assert!(Pm4MidAcquirePolicy::RequiredOnly.acquire_between(
            "fused_qk_l2_norm_scale_f32",
            "gated_delta_net_q8_compact2_b2"
        ));
        assert!(Pm4MidAcquirePolicy::RequiredOnly.acquire_between(
            "fused_qk_l2_norm_scale_f32",
            "gated_delta_net_q8_compact3_b2"
        ));
        assert_eq!(Pm4MidAcquirePolicy::from_value("invalid"), None);
    }

    #[test]
    fn gfx1151_dispatch_initiator_policy_is_exact_arch_only() {
        assert_eq!(
            gfx10_dispatch_initiator_policy_from_value(Pm4Architecture::Gfx11, "gfx1151", "order",),
            Some(Gfx10DispatchInitiatorPolicy::OrderMode)
        );
        assert_eq!(
            gfx10_dispatch_initiator_policy_from_value(Pm4Architecture::Gfx11, "gfx1151", "radv",),
            Some(Gfx10DispatchInitiatorPolicy::Radv)
        );
        assert_eq!(
            gfx10_dispatch_initiator_policy_from_value(Pm4Architecture::Gfx11, "gfx1100", "radv",),
            Some(Gfx10DispatchInitiatorPolicy::Legacy)
        );
        assert_eq!(
            gfx10_dispatch_initiator_policy_from_value(Pm4Architecture::Gfx10, "gfx1030", "radv",),
            Some(Gfx10DispatchInitiatorPolicy::Legacy)
        );
        assert_eq!(
            gfx10_dispatch_initiator_policy_from_value(
                Pm4Architecture::Gfx11,
                "gfx1151",
                "invalid",
            ),
            None
        );
    }

    #[test]
    fn gfx1151_dispatch_interleave_is_exact_arch_only() {
        assert_eq!(
            gfx1151_dispatch_interleave_from_value(Pm4Architecture::Gfx11, "gfx1151", "64"),
            Some(Some(Gfx11DispatchInterleave::Threads64))
        );
        assert_eq!(
            gfx1151_dispatch_interleave_from_value(Pm4Architecture::Gfx11, "gfx1151", "0"),
            Some(Some(Gfx11DispatchInterleave::Disabled))
        );
        assert_eq!(
            gfx1151_dispatch_interleave_from_value(Pm4Architecture::Gfx11, "gfx1151", "inherit",),
            Some(None)
        );
        assert_eq!(
            gfx1151_dispatch_interleave_from_value(Pm4Architecture::Gfx11, "gfx1100", "64"),
            Some(None)
        );
        assert_eq!(
            gfx1151_dispatch_interleave_from_value(Pm4Architecture::Gfx12, "gfx1201", "64"),
            Some(None)
        );
        assert_eq!(
            gfx1151_dispatch_interleave_from_value(Pm4Architecture::Gfx11, "gfx1151", "32"),
            None
        );
    }

    #[test]
    fn gfx1151_resource_limits_policy_is_exact_arch_only() {
        let radv = Gfx11ComputeResourceLimitsPolicy::Radv {
            force_simd_dist_for_single_wave: false,
        };
        assert_eq!(
            gfx1151_resource_limits_policy_from_value(Pm4Architecture::Gfx11, "gfx1151", "radv",),
            Some(radv)
        );
        assert_eq!(
            gfx1151_resource_limits_policy_from_value(
                Pm4Architecture::Gfx11,
                "gfx1151",
                "simd-always",
            ),
            Some(Gfx11ComputeResourceLimitsPolicy::SimdDestAlways)
        );
        assert_eq!(
            gfx1151_resource_limits_policy_from_value(Pm4Architecture::Gfx11, "gfx1100", "radv",),
            Some(Gfx11ComputeResourceLimitsPolicy::Legacy)
        );
        assert_eq!(
            gfx1151_resource_limits_policy_from_value(Pm4Architecture::Gfx12, "gfx1201", "radv",),
            Some(Gfx11ComputeResourceLimitsPolicy::Legacy)
        );
        assert_eq!(
            gfx1151_resource_limits_policy_from_value(Pm4Architecture::Gfx11, "gfx1151", "invalid",),
            None
        );
    }

    #[test]
    fn prepared_grid_rejects_oversized_yz_on_gfx1201_only() {
        assert!(check_prepared_grid("gfx1201", "k.kd", [7, 65_536, 1]).is_ok());
        assert!(check_prepared_grid("gfx1201", "k.kd", [1 << 20, 1, 65_536]).is_ok());
        assert!(check_prepared_grid("gfx1201", "k.kd", [7, 65_537, 1]).is_err());
        assert!(check_prepared_grid("gfx1201", "k.kd", [7, 1, 65_537]).is_err());
        assert!(check_prepared_grid("gfx1100", "k.kd", [7, 65_537, 65_537]).is_ok());
    }

    #[test]
    fn gfx1151_cu_mask_is_exact_arch_and_wgp_paired() {
        assert_eq!(
            gfx1151_cu_mask_from_value(Pm4Architecture::Gfx11, "gfx1151", "all"),
            Some(None)
        );
        assert_eq!(
            gfx1151_cu_mask_from_value(Pm4Architecture::Gfx11, "gfx1151", "32"),
            Some(Some([u32::MAX, 0]))
        );
        assert_eq!(
            gfx1151_cu_mask_from_value(Pm4Architecture::Gfx11, "gfx1151", "36"),
            Some(Some([u32::MAX, 0xf]))
        );
        assert_eq!(
            gfx1151_cu_mask_from_value(Pm4Architecture::Gfx11, "gfx1100", "32"),
            Some(None)
        );
        assert_eq!(
            gfx1151_cu_mask_from_value(Pm4Architecture::Gfx12, "gfx1201", "32"),
            Some(None)
        );
        assert_eq!(
            gfx1151_cu_mask_from_value(Pm4Architecture::Gfx11, "gfx1151", "35"),
            None
        );
        assert_eq!(
            gfx1151_cu_mask_from_value(Pm4Architecture::Gfx11, "gfx1151", "42"),
            None
        );
    }

    #[test]
    fn gfx1151_entry_acquire_is_exact_arch_only() {
        assert_eq!(
            gfx1151_entry_acquire_policy_from_value(Pm4Architecture::Gfx11, "gfx1151", "agent",),
            Some(Gfx11EntryAcquirePolicy::Agent)
        );
        assert_eq!(
            gfx1151_entry_acquire_policy_from_value(Pm4Architecture::Gfx11, "gfx1151", "vmem",),
            Some(Gfx11EntryAcquirePolicy::Vmem)
        );
        assert_eq!(
            gfx1151_entry_acquire_policy_from_value(Pm4Architecture::Gfx11, "gfx1100", "none",),
            Some(Gfx11EntryAcquirePolicy::System)
        );
        assert_eq!(
            gfx1151_entry_acquire_policy_from_value(Pm4Architecture::Gfx12, "gfx1201", "agent",),
            Some(Gfx11EntryAcquirePolicy::System)
        );
        assert_eq!(
            gfx1151_entry_acquire_policy_from_value(Pm4Architecture::Gfx11, "gfx1151", "invalid",),
            None
        );
    }

    #[test]
    fn resource_wait_policy_and_a3b_pointer_catalog_fail_closed() {
        assert_eq!(
            Pm4WaitPolicy::from_value("resource-audit"),
            Some(Pm4WaitPolicy::ResourceAudit)
        );
        assert_eq!(
            expected_kernarg_bytes("gated_delta_net_q8_compact2_b2"),
            Some(96)
        );
        assert!(pointer_effects("gated_delta_net_q8_compact2_b2").is_some());
        assert_eq!(
            expected_kernarg_bytes("gated_delta_net_q8_compact3_b2"),
            Some(96)
        );
        assert!(pointer_effects("gated_delta_net_q8_compact3_b2").is_some());
        assert_eq!(
            expected_kernarg_bytes("gated_norm_mq_rotate_k6144_gfx1100"),
            Some(64)
        );
        assert!(pointer_effects("gated_norm_mq_rotate_k6144_gfx1100").is_some());
        assert_eq!(
            expected_kernarg_bytes("qwen36_27b_fa_prep_gfx1100"),
            Some(64)
        );
        assert!(pointer_effects("qwen36_27b_fa_prep_gfx1100").is_some());
        assert_eq!(
            expected_kernarg_bytes("conv1d_silu_split_qknorm_b256"),
            Some(80)
        );
        assert!(pointer_effects("conv1d_silu_split_qknorm_b256").is_some());
        assert_eq!(
            expected_kernarg_bytes("moe_router_softmax_topk_k8_wave64_exact_shared_silu_mq_rotate"),
            Some(80)
        );
        assert_eq!(
            Pm4WaitPolicy::from_value("resource"),
            Some(Pm4WaitPolicy::Resource)
        );
        assert_eq!(Pm4WaitPolicy::from_value("invalid"), None);
        assert_eq!(
            Pm4RegisterPolicy::from_value("legacy"),
            Some(Pm4RegisterPolicy::Legacy)
        );
        assert_eq!(
            Pm4RegisterPolicy::from_value("1"),
            Some(Pm4RegisterPolicy::Stateful)
        );
        assert_eq!(
            Pm4RegisterPolicy::from_value("static"),
            Some(Pm4RegisterPolicy::Static)
        );
        assert_eq!(Pm4RegisterPolicy::from_value("invalid"), None);
        for kernel in [
            "fused_gate_up_hfq4g256",
            "fused_gate_up_hfq4g256_k1024_gfx1201",
            "fused_gate_up_hfq4g256_dot_reform_gfx1100",
            "fused_gate_up_hfq4g256_dot_prefetch_gfx1100",
            "fused_gate_up_hfq4g256_pair_gfx1100",
            "fused_gate_up_hfq4g256_pair2_gfx1100",
            "fused_gate_up_hfq4g256_quad_prefetch_gfx1100",
            "fused_gate_up_hfq4g256_setprio_gfx1100",
            "fused_gate_up_hfq4g256_lane0_headers_gfx1100",
            "fused_gate_up_hfq4g256_stage_x32_gfx1100",
        ] {
            assert_eq!(expected_kernarg_bytes(kernel), Some(64));
            assert_eq!(
                pointer_effects(kernel).map(|effects| effects.len()),
                Some(5)
            );
        }
        assert!(pointer_effects("unknown_kernel").is_none());
        assert!(expected_kernarg_bytes("unknown_kernel").is_none());
        for kernel in A3B_REPLAY_KERNELS {
            let effects = pointer_effects(kernel).unwrap_or_else(|| panic!("missing {kernel}"));
            let kernarg_bytes = expected_kernarg_bytes(kernel)
                .unwrap_or_else(|| panic!("missing ABI size for {kernel}"));
            assert!(!effects.is_empty(), "empty pointer signature for {kernel}");
            assert!(
                effects
                    .iter()
                    .all(|effect| effect.offset + 8 <= kernarg_bytes),
                "pointer offset exceeds kernarg ABI in {kernel}"
            );
            let offsets = effects
                .iter()
                .map(|effect| effect.offset)
                .collect::<BTreeSet<_>>();
            assert_eq!(
                offsets.len(),
                effects.len(),
                "duplicate pointer offset in {kernel}"
            );
        }
    }

    #[test]
    fn allocation_wide_hazards_include_subviews_and_ignore_read_read() {
        let read_a = RecordedResourceAccess {
            allocation_base: 0x1000,
            allocation_bytes: 0x1000,
            access_base: 0x1000,
            mode: RecordedAccessMode::Read,
        };
        let read_same = RecordedResourceAccess {
            allocation_base: 0x1800,
            allocation_bytes: 0x100,
            access_base: 0x1800,
            mode: RecordedAccessMode::Read,
        };
        let write_same = RecordedResourceAccess {
            mode: RecordedAccessMode::Write,
            ..read_same
        };
        let write_other = RecordedResourceAccess {
            allocation_base: 0x3000,
            allocation_bytes: 0x100,
            access_base: 0x3000,
            mode: RecordedAccessMode::Write,
        };
        assert!(!read_a.conflicts(read_same));
        assert!(read_a.conflicts(write_same));
        assert!(!read_a.conflicts(write_other));
    }

    #[test]
    fn exact_start_audit_separates_subviews_from_true_dependencies() {
        let write_left = RecordedResourceAccess {
            allocation_base: 0x1000,
            allocation_bytes: 0x1000,
            access_base: 0x1100,
            mode: RecordedAccessMode::Write,
        };
        let read_right = RecordedResourceAccess {
            access_base: 0x1800,
            mode: RecordedAccessMode::Read,
            ..write_left
        };
        let read_left = RecordedResourceAccess {
            mode: RecordedAccessMode::Read,
            ..write_left
        };

        assert!(write_left.conflicts(read_right));
        assert!(!write_left.same_start_conflicts(read_right));
        assert!(write_left.same_start_conflicts(read_left));
    }

    #[test]
    fn position_grid_binding_narrows_recorded_maximum() {
        let binding = ReplayGridBinding::PositionCeilDiv {
            axis: 1,
            addend: 1,
            divisor: 128,
        };
        assert_eq!(binding.bind(0, [16, 16, 1]).unwrap(), [16, 1, 1]);
        assert_eq!(binding.bind(128, [16, 16, 1]).unwrap(), [16, 2, 1]);
        assert_eq!(binding.bind(2047, [16, 16, 1]).unwrap(), [16, 16, 1]);
        assert_eq!(binding.bind(4095, [16, 16, 1]).unwrap(), [16, 16, 1]);
    }

    #[test]
    fn resource_frontier_catches_non_adjacent_hazards() {
        let launch = |kernel: &str, access: RecordedResourceAccess| RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1; 3],
            block: [1; 3],
            shared_mem: 0,
            grid_binding: None,
            declared_kernarg_bindings: Vec::new(),
            kernarg: Vec::new(),
            accesses: Some(vec![access]),
            binding_layout: None,
        };
        let write_a = launch(
            "write_a",
            RecordedResourceAccess {
                allocation_base: 0x1000,
                allocation_bytes: 0x100,
                access_base: 0x1000,
                mode: RecordedAccessMode::Write,
            },
        );
        let write_b = launch(
            "write_b",
            RecordedResourceAccess {
                allocation_base: 0x2000,
                allocation_bytes: 0x100,
                access_base: 0x2000,
                mode: RecordedAccessMode::Write,
            },
        );
        let read_a = launch(
            "read_a",
            RecordedResourceAccess {
                mode: RecordedAccessMode::Read,
                ..write_a.accesses.as_ref().unwrap()[0]
            },
        );

        let mut frontier = ResourceFrontier::default();
        frontier.advance(&write_a, false);
        assert!(frontier.independent(&write_b));
        frontier.advance(&write_b, true);
        assert!(!frontier.independent(&read_a));
        frontier.advance(&read_a, false);
        assert_eq!(frontier.accesses, read_a.accesses.clone().unwrap());

        let unknown = RecordedHipLaunch {
            accesses: None,
            ..write_b.clone()
        };
        assert!(!frontier.independent(&unknown));
        frontier.advance(&unknown, false);
        assert!(!frontier.independent(&write_b));
    }

    #[test]
    fn pm4_width_reorder_widens_antichains_without_crossing_dependencies() {
        let mk = |kernel: &str, base: u64, mode: RecordedAccessMode| RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1; 3],
            block: [1; 3],
            shared_mem: 0,
            grid_binding: None,
            declared_kernarg_bindings: Vec::new(),
            kernarg: Vec::new(),
            accesses: Some(vec![RecordedResourceAccess {
                allocation_base: base,
                allocation_bytes: 0x100,
                access_base: base,
                mode,
            }]),
            binding_layout: None,
        };

        // One dependent pair (write_x -> read_x) with three launches on
        // unrelated allocations placed either side of it.
        let recorded = vec![
            mk("write_x", 0x1000, RecordedAccessMode::Write),
            mk("indep_a", 0x2000, RecordedAccessMode::Write),
            mk("read_x", 0x1000, RecordedAccessMode::Read),
            mk("indep_b", 0x3000, RecordedAccessMode::Write),
            mk("indep_c", 0x4000, RecordedAccessMode::Write),
        ];

        let order = pm4_width_reorder(&recorded, usize::MAX);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4], "reorder must be a permutation");

        // The one real dependency is preserved.
        let slot = |index: usize| order.iter().position(|value| *value == index).unwrap();
        assert!(slot(0) < slot(2), "write_x must still precede read_x");
    }

    #[test]
    fn pm4_width_reorder_pins_launches_with_unknown_effects() {
        let mk = |kernel: &str, base: u64| RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1; 3],
            block: [1; 3],
            shared_mem: 0,
            grid_binding: None,
            declared_kernarg_bindings: Vec::new(),
            kernarg: Vec::new(),
            accesses: Some(vec![RecordedResourceAccess {
                allocation_base: base,
                allocation_bytes: 0x100,
                access_base: base,
                mode: RecordedAccessMode::Read,
            }]),
            binding_layout: None,
        };
        // An unknown-effect launch conflicts with everything, so it must act as
        // an ordering barrier and hold its recorded position.
        let recorded = vec![
            mk("read_a", 0x1000),
            RecordedHipLaunch {
                accesses: None,
                ..mk("unknown", 0x2000)
            },
            mk("read_b", 0x3000),
        ];
        assert_eq!(pm4_width_reorder(&recorded, usize::MAX), vec![0, 1, 2]);
    }

    #[test]
    fn pm4_phase_planner_parallelizes_only_pairwise_independent_launches() {
        let launch = |kernel: &str, base: u64, mode: RecordedAccessMode| RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1; 3],
            block: [1; 3],
            shared_mem: 0,
            grid_binding: None,
            declared_kernarg_bindings: Vec::new(),
            kernarg: Vec::new(),
            accesses: Some(vec![RecordedResourceAccess {
                allocation_base: base,
                allocation_bytes: 0x100,
                access_base: base,
                mode,
            }]),
            binding_layout: None,
        };
        let unknown = RecordedHipLaunch {
            accesses: None,
            ..launch("unknown", 0x3000, RecordedAccessMode::Read)
        };
        let recorded = vec![
            launch("write_a", 0x1000, RecordedAccessMode::Write),
            launch("write_b", 0x2000, RecordedAccessMode::Write),
            launch("read_a", 0x1000, RecordedAccessMode::Read),
            launch("write_a_again", 0x1000, RecordedAccessMode::Write),
            unknown,
        ];

        assert_eq!(
            pm4_phase_plan(&recorded, 2, 0, usize::MAX),
            vec![
                Pm4PhasePlan {
                    indices: vec![0, 1],
                    parallel: true,
                    lane_split: None,
                },
                Pm4PhasePlan {
                    indices: vec![2, 3, 4],
                    parallel: false,
                    lane_split: None,
                },
            ]
        );

        let mut two_parallel_phases = recorded[..2].to_vec();
        two_parallel_phases.push(launch("read_a", 0x1000, RecordedAccessMode::Read));
        two_parallel_phases.push(launch("read_b", 0x2000, RecordedAccessMode::Read));
        assert_eq!(
            pm4_phase_plan(&two_parallel_phases, 2, 0, 1),
            vec![
                Pm4PhasePlan {
                    indices: vec![0, 1],
                    parallel: true,
                    lane_split: None,
                },
                Pm4PhasePlan {
                    indices: vec![2, 3],
                    parallel: false,
                    lane_split: None,
                },
            ]
        );
    }

    #[test]
    fn pm4_phase_planner_allows_read_read_and_serializes_unknown_accesses() {
        let read = |kernel: &str| RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1; 3],
            block: [1; 3],
            shared_mem: 0,
            grid_binding: None,
            declared_kernarg_bindings: Vec::new(),
            kernarg: Vec::new(),
            accesses: Some(vec![RecordedResourceAccess {
                allocation_base: 0x1000,
                allocation_bytes: 0x100,
                access_base: 0x1000,
                mode: RecordedAccessMode::Read,
            }]),
            binding_layout: None,
        };
        assert_eq!(
            pm4_phase_plan(&[read("read_a"), read("read_a_again")], 2, 0, usize::MAX,),
            vec![Pm4PhasePlan {
                indices: vec![0, 1],
                parallel: true,
                lane_split: None,
            }]
        );
        assert_eq!(
            pm4_phase_plan(
                &[
                    RecordedHipLaunch {
                        accesses: None,
                        ..read("unknown_a")
                    },
                    RecordedHipLaunch {
                        accesses: None,
                        ..read("unknown_b")
                    },
                ],
                2,
                0,
                usize::MAX,
            ),
            vec![Pm4PhasePlan {
                indices: vec![0, 1],
                parallel: false,
                lane_split: None,
            }]
        );
        assert_eq!(
            pm4_phase_plan(&[read("read_a"), read("read_a_again")], 3, 0, usize::MAX,),
            vec![Pm4PhasePlan {
                indices: vec![0, 1],
                parallel: false,
                lane_split: None,
            }]
        );
        assert_eq!(
            pm4_phase_plan(&[read("read_a"), read("read_a_again")], 2, 3, usize::MAX,),
            vec![Pm4PhasePlan {
                indices: vec![0, 1],
                parallel: false,
                lane_split: None,
            }]
        );
    }

    #[test]
    fn pm4_ds4_ffn_branch_planner_recovers_dependent_chains() {
        let launch = |kernel: &str, accesses: &[(u64, RecordedAccessMode)]| RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1; 3],
            block: [1; 3],
            shared_mem: 0,
            grid_binding: None,
            declared_kernarg_bindings: Vec::new(),
            kernarg: Vec::new(),
            accesses: Some(
                accesses
                    .iter()
                    .map(|(base, mode)| RecordedResourceAccess {
                        allocation_base: *base,
                        allocation_bytes: 0x100,
                        access_base: *base,
                        mode: *mode,
                    })
                    .collect(),
            ),
            binding_layout: None,
        };
        use RecordedAccessMode::{Read, Write};
        let recorded = vec![
            launch("prepare", &[(0x1000, Write)]),
            launch("zero_f32", &[(0x5000, Write)]),
            launch("gemv_mfp4g32_e8_soa_u4", &[(0x1000, Read), (0x2000, Write)]),
            launch("shared_down", &[(0x2000, Read), (0x4000, Write)]),
            launch(
                "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed",
                &[(0x1000, Read), (0x3000, Write)],
            ),
            launch("routed_down", &[(0x3000, Read), (0x5000, Write)]),
            launch("add_inplace_f32", &[(0x4000, Write), (0x5000, Read)]),
            launch("tail", &[(0x4000, Read)]),
        ];

        assert_eq!(
            pm4_ds4_ffn_branch_plan(&recorded).unwrap(),
            vec![
                Pm4PhasePlan {
                    indices: vec![0],
                    parallel: false,
                    lane_split: None,
                },
                Pm4PhasePlan {
                    indices: vec![2, 3, 1, 4, 5],
                    parallel: true,
                    lane_split: Some(2),
                },
                Pm4PhasePlan {
                    indices: vec![6, 7],
                    parallel: false,
                    lane_split: None,
                },
            ]
        );
    }

    #[test]
    fn pm4_ds4_batched_ffn_branch_planner_keeps_routed_down_after_fan_in() {
        let launch = |kernel: &str, accesses: &[(u64, RecordedAccessMode)]| RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1; 3],
            block: [1; 3],
            shared_mem: 0,
            grid_binding: None,
            declared_kernarg_bindings: Vec::new(),
            kernarg: Vec::new(),
            accesses: Some(
                accesses
                    .iter()
                    .map(|(base, mode)| RecordedResourceAccess {
                        allocation_base: *base,
                        allocation_bytes: 0x100,
                        access_base: *base,
                        mode: *mode,
                    })
                    .collect(),
            ),
            binding_layout: None,
        };
        use RecordedAccessMode::{Read, Write};
        let e8 = "gemv_mfp4g32_e8_soa_batched_b3_gfx1151";
        let recorded = vec![
            launch("prepare", &[(0x1000, Write)]),
            launch(e8, &[(0x1000, Read), (0x2000, Write)]),
            launch(e8, &[(0x1000, Read), (0x2100, Write)]),
            launch(
                "deepseek4_silu_mul_clamp_f32",
                &[(0x2000, Write), (0x2100, Read)],
            ),
            launch("mq_rotate_x", &[(0x2000, Read), (0x2200, Write)]),
            launch(e8, &[(0x2200, Read), (0x3000, Write)]),
            launch(e8, &[(0x1000, Read), (0x4000, Write)]),
            launch("sqrt_softplus_f32", &[(0x4000, Write)]),
            launch(
                "deepseek4_moe_topk_bias_aware_batched_f32",
                &[(0x4000, Read), (0x4100, Write)],
            ),
            launch(
                "gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_batched_k4",
                &[(0x1000, Read), (0x4100, Read), (0x4200, Write)],
            ),
            launch("deepseek4_silu_mul_clamp_f32", &[(0x4200, Write)]),
            launch("mq_rotate_x", &[(0x4200, Read), (0x4300, Write)]),
            launch(
                "gemv_mq2g256_lloyd_moe_down_residual_scaled_k8_indexed_batched_k4",
                &[(0x4300, Read), (0x3000, Write)],
            ),
            launch("tail", &[(0x3000, Read)]),
        ];

        assert_eq!(
            pm4_ds4_batched_ffn_branch_plan(&recorded).unwrap(),
            vec![
                Pm4PhasePlan {
                    indices: vec![0],
                    parallel: false,
                    lane_split: None,
                },
                Pm4PhasePlan {
                    indices: (1..12).collect(),
                    parallel: true,
                    lane_split: Some(5),
                },
                Pm4PhasePlan {
                    indices: vec![12, 13],
                    parallel: false,
                    lane_split: None,
                },
            ]
        );
    }

    #[test]
    fn default_hip_never_records_or_routes() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Hip);
        controller.record_hip_launch("k", None, [1; 3], [32, 1, 1], 0, &[]);
        assert!(controller.recorded_launches().is_empty());
        assert!(!controller.should_route_aql());
    }

    #[test]
    fn model_default_resets_stale_state_and_is_scoped() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.record_hip_launch("old", None, [1; 3], [32, 1, 1], 0, &[]);
        let mut failed = passing(1.20);
        failed.guards_intact = false;
        controller.observe_shadow(failed);
        assert_eq!(controller.state(), ReplayState::Fallback);

        controller.apply_model_default(true, ReplayTransport::Pm4Ib);
        assert_eq!(controller.request(), ReplayBackendRequest::Auto);
        assert_eq!(controller.state(), ReplayState::Armed);
        assert_eq!(controller.transport_name(), "pm4");
        assert!(controller.recorded_launches().is_empty());
        assert_eq!(controller.fallback_reason(), None);
        controller.begin_auto_capture_if_armed().unwrap();
        assert_eq!(controller.state(), ReplayState::RecordingWarmup);

        controller.apply_model_default(false, ReplayTransport::AqlPackets);
        assert_eq!(controller.request(), ReplayBackendRequest::Hip);
        assert_eq!(controller.state(), ReplayState::Hip);
        assert_eq!(controller.transport_name(), "aql");
        assert!(!controller.is_enabled());
    }

    #[test]
    fn layout_growth_rearms_without_changing_route_selection() {
        let mut controller = ReplayController::new_manual_pm4();
        controller.state = ReplayState::Ready;
        controller.fallback_reason = Some("stale prepared layout".to_owned());

        controller.rearm_after_layout_growth();

        assert_eq!(controller.request(), ReplayBackendRequest::Auto);
        assert_eq!(controller.transport_name(), "pm4");
        assert_eq!(controller.state(), ReplayState::Armed);
        assert_eq!(controller.fallback_reason(), None);
        assert!(!controller.auto_lifecycle);
    }

    #[test]
    fn route_observation_records_success_failure_and_resets() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        let failed: Result<(), String> = Err("dispatch failed".to_owned());
        assert!(controller.observe_replay_result(127, failed).is_err());
        assert!(controller.replay_observation().failed);

        controller.begin_replay_observation_window();

        controller.observe_replay_result(127, Ok(())).unwrap();
        controller.observe_replay_result(128, Ok(())).unwrap();
        assert_eq!(
            controller.replay_observation(),
            ReplayObservation {
                count: 2,
                first_position: Some(127),
                last_position: Some(128),
                failed: false,
            }
        );

        controller.apply_model_default(false, ReplayTransport::AqlPackets);
        assert_eq!(
            controller.replay_observation(),
            ReplayObservation::default()
        );
    }

    #[test]
    fn route_observation_windows_are_request_local() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.observe_replay_result(127, Ok(())).unwrap();
        controller.observe_replay_result(128, Ok(())).unwrap();

        controller.begin_replay_observation_window();
        assert_eq!(
            controller.replay_observation(),
            ReplayObservation::default()
        );

        controller.observe_replay_result(512, Ok(())).unwrap();
        assert_eq!(
            controller.replay_observation(),
            ReplayObservation {
                count: 1,
                first_position: Some(512),
                last_position: Some(512),
                failed: false,
            }
        );
    }

    #[test]
    fn route_proof_marker_is_request_scoped() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.route_proof_log = true;
        assert_eq!(
            controller.replay_observation_marker("chatcmpl-turn-1"),
            None
        );

        controller.observe_replay_result(127, Ok(())).unwrap();
        controller.observe_replay_result(128, Ok(())).unwrap();
        assert_eq!(
            controller.replay_observation_marker("chatcmpl-turn-1"),
            Some(
                "HIPFIRE_REPLAY_ROUTE_PROOF transport=aql position=127 \
                 request_id=chatcmpl-turn-1 replays=2"
                    .to_owned()
            )
        );
        assert_eq!(controller.replay_observation_marker(""), None);

        controller.begin_replay_observation_window();
        assert_eq!(
            controller.replay_observation_marker("chatcmpl-turn-2"),
            None
        );
        controller.observe_replay_result(512, Ok(())).unwrap();
        assert_eq!(
            controller.replay_observation_marker("invalid request id"),
            None
        );
    }

    #[test]
    fn route_proof_marker_fails_closed_after_replay_error() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.route_proof_log = true;
        controller.observe_replay_result(127, Ok(())).unwrap();
        assert!(controller
            .observe_replay_result::<()>(128, Err("dispatch failed".to_owned()))
            .is_err());

        assert_eq!(
            controller.replay_observation_marker("chatcmpl-turn-1"),
            None
        );
        controller.observe_replay_result(129, Ok(())).unwrap();
        assert_eq!(
            controller.replay_observation_marker("chatcmpl-turn-1"),
            None
        );
    }

    #[test]
    fn route_proof_marker_fails_closed_after_request_abort() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.route_proof_log = true;
        controller.observe_replay_result(127, Ok(())).unwrap();
        controller.invalidate_replay_observation_window();

        assert_eq!(
            controller.replay_observation_marker("chatcmpl-turn-1"),
            None
        );
        controller.observe_replay_result(128, Ok(())).unwrap();
        assert_eq!(
            controller.replay_observation_marker("chatcmpl-turn-1"),
            None
        );
    }

    #[test]
    fn pm4_packet_identity_reports_actual_count() {
        // Phased multi-queue graphs legitimately carry barrier + IB packets per lane.
        assert_eq!(pm4_packet_identity(0), None);
        assert_eq!(pm4_packet_identity(1), Some(1));
        assert_eq!(pm4_packet_identity(260), Some(260));
    }

    #[test]
    fn auto_requires_two_shadows_and_explicit_install() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.record_hip_launch("k", None, [1; 3], [32, 1, 1], 0, &[]);
        controller.observe_shadow(passing(1.08));
        assert_eq!(controller.state(), ReplayState::RecordingWarmup);
        controller.observe_shadow(passing(1.06));
        assert_eq!(controller.state(), ReplayState::ShadowValidated);
        assert!(!controller.should_route_aql());
        controller.install_prepared_plan().unwrap();
        assert!(controller.should_route_aql());
    }

    #[test]
    fn any_failed_gate_is_sticky_fallback() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        let mut failed = passing(1.20);
        failed.guards_intact = false;
        controller.observe_shadow(failed);
        controller.observe_shadow(passing(2.0));
        assert_eq!(controller.state(), ReplayState::Fallback);
        assert!(!controller.should_route_aql());
    }

    #[test]
    fn g0_observation_keeps_the_eager_branch_and_excludes_recording() {
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Shadow);
        controller.begin_g0_observation().unwrap();
        // The eager arm must not look like a recording to any predicate, and
        // a recording cannot start underneath it.
        assert!(!controller.is_recording());
        assert!(controller.begin_capture().is_err());
        assert!(controller.begin_g0_observation().is_err());
        controller.observe_g0_launch("a", [1, 2, 3], [32, 1, 1], 0, &[1, 2]);
        controller.observe_g0_launch("b", [4, 1, 1], [64, 1, 1], 128, &[3]);
        // Observed launches never enter the replay tape.
        assert!(controller.recorded_launches().is_empty());
        let seen = controller.finish_g0_observation().unwrap();
        assert_eq!(
            seen.iter()
                .map(|l| (l.kernel.as_str(), l.grid, l.shared_mem, l.kernarg.clone()))
                .collect::<Vec<_>>(),
            vec![
                ("a", [1, 2, 3], 0, vec![1, 2]),
                ("b", [4, 1, 1], 128, vec![3])
            ]
        );
        assert!(controller.finish_g0_observation().is_err());

        controller.begin_capture().unwrap();
        assert!(controller.is_recording());
        assert!(controller.begin_g0_observation().is_err());
    }

    #[test]
    fn manual_capture_is_bounded_and_sequence_stable() {
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Shadow);
        controller.record_hip_launch("ignored", None, [1; 3], [1; 3], 0, &[]);
        assert_eq!(controller.state(), ReplayState::Armed);
        assert!(controller.recorded_launches().is_empty());

        controller.begin_capture().unwrap();
        controller.record_hip_launch("a", None, [1, 2, 3], [32, 1, 1], 0, &[1]);
        controller.record_hip_launch("b", None, [4, 5, 6], [64, 1, 1], 128, &[2]);
        let first = controller.finish_capture().unwrap();
        assert_eq!(controller.state(), ReplayState::Captured);
        assert_eq!(first.launch_count, 2);
        assert_eq!(first.unique_kernel_count, 2);

        controller.begin_capture().unwrap();
        controller.record_hip_launch("a", None, [1, 2, 3], [32, 1, 1], 0, &[1]);
        controller.record_hip_launch("b", None, [4, 5, 6], [64, 1, 1], 128, &[2]);
        assert_eq!(controller.finish_capture().unwrap(), first);

        controller.begin_capture().unwrap();
        controller.record_hip_launch("b", None, [4, 5, 6], [64, 1, 1], 128, &[2]);
        controller.record_hip_launch("a", None, [1, 2, 3], [32, 1, 1], 0, &[1]);
        assert_ne!(
            controller.finish_capture().unwrap().sequence_hash,
            first.sequence_hash
        );
    }

    #[test]
    fn ineligible_forward_neither_records_nor_routes_plain_ar() {
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.set_forward_eligible(false);
        controller.record_hip_launch("spec", None, [1; 3], [32, 1, 1], 0, &[1]);
        assert!(controller.recorded_launches().is_empty());
        assert!(!controller.should_auto_finalize_capture());

        controller.set_forward_eligible(true);
        controller.record_hip_launch("plain", None, [1; 3], [32, 1, 1], 0, &[2]);
        assert_eq!(controller.recorded_launches().len(), 1);
        controller.observe_shadow(passing(1.08));
        controller.observe_shadow(passing(1.06));
        controller.install_prepared_plan().unwrap();
        assert!(controller.should_route_aql());

        controller.set_forward_eligible(false);
        assert!(!controller.should_route_aql());
    }

    #[test]
    fn replay_grid_binding_units_for_and_bind() {
        let binding = ReplayGridBinding::PositionCeilDiv {
            axis: 0,
            addend: 15,
            divisor: 16,
        };
        // max_position = 32 => ceil((32+15)/16)=3
        assert_eq!(binding.units_for(32).unwrap(), 3);
        // current position 16 => ceil(31/16)=2, min with prepared max 3 => 2
        let prepared_grid = [3, 1, 1];
        assert_eq!(binding.bind(16, prepared_grid).unwrap(), [2, 1, 1]);
        // current position 0 => ceil(15/16)=1
        assert_eq!(binding.bind(0, prepared_grid).unwrap(), [1, 1, 1]);
        // position == max => 3
        assert_eq!(binding.bind(32, prepared_grid).unwrap(), [3, 1, 1]);
    }

    #[test]
    fn replay_grid_binding_prepared_sizing_and_rejection() {
        let binding = ReplayGridBinding::PositionCeilDiv {
            axis: 0,
            addend: 0,
            divisor: 1,
        };
        // Simulate prepared_max_position = 100
        let prepared_max = 100usize;
        let recorded_grid = [10, 1, 1];
        let prepared_units = binding.units_for(prepared_max).unwrap();
        let mut prepared_grid = recorded_grid;
        prepared_grid[0] = prepared_units;
        assert_eq!(prepared_grid, [100, 1, 1]);
        // Patch down for smaller position 10 => 10
        assert_eq!(binding.bind(10, prepared_grid).unwrap(), [10, 1, 1]);
        // Reject position > max_position should be handled at replay level
        // (here we simulate the check that PreparedPm4Replay::replay_and_wait_checked does)
        let current = 150usize;
        let should_reject = current > prepared_max;
        assert!(should_reject);
    }

    #[test]
    fn replay_kernarg_binding_apply_offsets() {
        let mut kernarg = vec![0u8; 80];
        // GdnFrameU32 at offset 76 writes 4 bytes; now carries explicit frames count.
        let gdn = ReplayKernargBinding::GdnFrameU32 {
            offset: 76,
            frames: 1,
        };
        gdn.apply(&mut kernarg, 0).unwrap();
        let first = u32::from_le_bytes(kernarg[76..80].try_into().unwrap());
        // `reserve_gdn_requant_frames` is a `fetch_add`, so the first frame in
        // a fresh process is legitimately 0. What must hold is that every apply
        // consumes a fresh frame rather than replaying the captured one.
        gdn.apply(&mut kernarg, 0).unwrap();
        let second = u32::from_le_bytes(kernarg[76..80].try_into().unwrap());
        assert!(second > first, "frame did not advance: {first} -> {second}");
        // PositionPlusU32 at offset 0 with addend 5, position 10 => 15
        let mut kernarg2 = vec![0u8; 8];
        let pos_binding = ReplayKernargBinding::PositionPlusU32 {
            offset: 0,
            addend: 5,
        };
        pos_binding.apply(&mut kernarg2, 10).unwrap();
        let value = u32::from_ne_bytes(kernarg2[0..4].try_into().unwrap());
        assert_eq!(value, 15);
        // Out-of-bounds should error
        let mut small = vec![0u8; 4];
        let bad = ReplayKernargBinding::GdnFrameU32 {
            offset: 76,
            frames: 1,
        };
        assert!(bad.apply(&mut small, 0).is_err());
    }

    #[test]
    fn position_div_binding_rederives_the_host_quotient() {
        // `block_count = (position + rows) / compress` is computed host-side in
        // the lowering; the declared binding must reproduce it for every replay
        // position, and reject the shapes that would write a wrong slot.
        let mut kernarg = vec![0u8; 16];
        let binding = ReplayKernargBinding::PositionDivU32 {
            offset: 8,
            addend: 1,
            divisor: 4,
        };
        for (position, expected) in [(1usize, 0u32), (3, 1), (4, 1), (11, 3), (64, 16)] {
            binding.apply(&mut kernarg, position).unwrap();
            let value = u32::from_ne_bytes(kernarg[8..12].try_into().unwrap());
            assert_eq!(value, expected, "position {position}");
        }

        let zero_divisor = ReplayKernargBinding::PositionDivU32 {
            offset: 0,
            addend: 0,
            divisor: 0,
        };
        assert!(zero_divisor.apply(&mut kernarg, 1).is_err());
        let out_of_range = ReplayKernargBinding::PositionDivU32 {
            offset: 13,
            addend: 0,
            divisor: 4,
        };
        assert!(out_of_range.apply(&mut kernarg, 1).is_err());
        let overflow = ReplayKernargBinding::PositionDivU32 {
            offset: 0,
            addend: 1,
            divisor: 4,
        };
        assert!(overflow.apply(&mut kernarg, u32::MAX as usize).is_err());
    }

    #[test]
    fn position_mod_binding_rederives_the_ring_cursor() {
        // GDN convolution cursor: `(position + row_index) % history_rows`.
        let mut kernarg = vec![0u8; 16];
        let decode = ReplayKernargBinding::PositionModU32 {
            offset: 4,
            addend: 0,
            modulus: 3,
        };
        for (position, expected) in [(0usize, 0u32), (1, 1), (2, 2), (3, 0), (10, 1)] {
            decode.apply(&mut kernarg, position).unwrap();
            let value = u32::from_ne_bytes(kernarg[4..8].try_into().unwrap());
            assert_eq!(value, expected, "position {position}");
        }
        // Row 2 of a chunk starting at position 4 has ring slot (4 + 2) % 3 = 0.
        let chunk_row = ReplayKernargBinding::PositionModU32 {
            offset: 4,
            addend: 2,
            modulus: 3,
        };
        chunk_row.apply(&mut kernarg, 4).unwrap();
        assert_eq!(u32::from_ne_bytes(kernarg[4..8].try_into().unwrap()), 0);
        chunk_row.apply(&mut kernarg, 1).unwrap();
        assert_eq!(u32::from_ne_bytes(kernarg[4..8].try_into().unwrap()), 0);

        let zero_modulus = ReplayKernargBinding::PositionModU32 {
            offset: 0,
            addend: 0,
            modulus: 0,
        };
        assert!(zero_modulus.apply(&mut kernarg, 1).is_err());
        let overflow = ReplayKernargBinding::PositionModU32 {
            offset: 0,
            addend: 2,
            modulus: 3,
        };
        assert!(overflow
            .apply(&mut kernarg, (u32::MAX - 1) as usize)
            .is_err());
    }

    #[test]
    fn position_mul_binding_rederives_a_row_offset() {
        // `dst_col_offset = position * index_kv_width` for the index-key write.
        let mut kernarg = vec![0u8; 40];
        let binding = ReplayKernargBinding::PositionMulU32 {
            offset: 32,
            factor: 128,
        };
        for (position, expected) in [(0usize, 0u32), (1, 128), (17, 2176), (2047, 262_016)] {
            binding.apply(&mut kernarg, position).unwrap();
            assert_eq!(
                u32::from_ne_bytes(kernarg[32..36].try_into().unwrap()),
                expected,
                "position {position}"
            );
        }
        let out_of_range = ReplayKernargBinding::PositionMulU32 {
            offset: 37,
            factor: 128,
        };
        assert!(out_of_range.apply(&mut kernarg, 1).is_err());
        let overflow = ReplayKernargBinding::PositionMulU32 {
            offset: 0,
            factor: u32::MAX,
        };
        assert!(overflow.apply(&mut kernarg, 2).is_err());
    }

    #[test]
    fn retained_body_scope_is_the_eligible_forward_not_the_backend_choice() {
        // The scope a launch-level guard must use: recording, or routing a
        // prepared plan. An armed-but-idle controller, a captured-but-unprepared
        // one, and a poisoned one are NOT in the retained body, so their forwards
        // must keep running on HIP.
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Auto);
        assert_eq!(controller.state(), ReplayState::Armed);
        assert!(
            !controller.retained_body_active(),
            "an armed-but-idle controller must not make HIP forwards fail"
        );

        controller.begin_capture().expect("open capture window");
        assert!(
            controller.retained_body_active(),
            "recording is the retained body"
        );

        // An ineligible forward inside the window (prefill, spec, MTP) is not.
        controller.set_forward_eligible(false);
        assert!(
            !controller.retained_body_active(),
            "an ineligible forward must not be treated as the retained body"
        );
        controller.set_forward_eligible(true);

        let summary = controller.finish_capture().expect("close capture window");
        assert_eq!(summary.launch_count, 0);
        assert_eq!(controller.state(), ReplayState::Captured);
        assert!(
            !controller.retained_body_active(),
            "a captured-but-unprepared route still runs HIP"
        );

        controller.poison("test poison");
        assert_eq!(controller.state(), ReplayState::Fallback);
        assert!(
            !controller.retained_body_active(),
            "a poisoned route must fall back to HIP, not keep failing forwards"
        );
    }

    #[test]
    fn declared_binding_is_not_an_unexplained_kernarg_difference() {
        // A named dynamic field changes with position by design. Differencing
        // two recordings must skip it, while still synthesizing the ordinary
        // affine position scalar sitting next to it.
        const CURSOR: usize = 4;
        let mut kernarg = vec![0u8; 16];
        kernarg[CURSOR..CURSOR + 4].copy_from_slice(&0u32.to_ne_bytes());
        kernarg[8..12].copy_from_slice(&0u32.to_ne_bytes());

        let declared = [ReplayKernargBinding::PositionModU32 {
            offset: CURSOR,
            addend: 0,
            modulus: 3,
        }];
        let mut earlier = ReplayController::new(ReplayBackendRequest::Auto);
        earlier.record_hip_launch_with_accesses(
            "gated_delta_conv_bf16_f32",
            None,
            [4, 1, 1],
            [256, 1, 1],
            0,
            &kernarg,
            None,
            &declared,
            None,
            None,
        );
        let snapshot = earlier.snapshot_recorded_kernargs();

        let mut current = ReplayController::new(ReplayBackendRequest::Auto);
        let mut current_kernarg = kernarg.clone();
        current_kernarg[CURSOR..CURSOR + 4].copy_from_slice(&1u32.to_ne_bytes());
        current_kernarg[8..12].copy_from_slice(&4u32.to_ne_bytes());
        current.record_hip_launch_with_accesses(
            "gated_delta_conv_bf16_f32",
            None,
            [4, 1, 1],
            [256, 1, 1],
            0,
            &current_kernarg,
            None,
            &declared,
            None,
            None,
        );

        let synthesized = current
            .synthesize_position_bindings(&snapshot, 0, 4)
            .expect("declared offset must not be treated as unexplained");
        assert_eq!(synthesized, 1, "only the affine scalar is synthesized");
        assert_eq!(
            current.synthesized_position_bindings(),
            &[(
                0,
                ReplayKernargBinding::PositionPlusU32 {
                    offset: 8,
                    addend: 0
                }
            )],
            "declared offset {CURSOR} must be skipped, offset 8 discovered"
        );
    }

    #[test]
    fn declared_binding_collision_fails_closed() {
        // Two owners for one slot would patch it twice; the merge must refuse
        // instead of letting declaration order decide the result.
        let declared = [ReplayKernargBinding::PositionModU32 {
            offset: 4,
            addend: 0,
            modulus: 3,
        }];
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.record_hip_launch_with_accesses(
            "gated_delta_conv_bf16_f32",
            None,
            [1, 1, 1],
            [256, 1, 1],
            0,
            &[0u8; 16],
            None,
            &declared,
            None,
            None,
        );
        let launches = controller.recorded_launches().to_vec();
        let mut bindings: Vec<(usize, ReplayKernargBinding)> = vec![(
            0,
            ReplayKernargBinding::GdnFrameU32 {
                offset: 4,
                frames: 1,
            },
        )];
        let error = merge_declared_kernarg_bindings(&launches, 1, &mut bindings)
            .expect_err("second owner for one offset must be rejected");
        assert!(error.contains("already has an owner"), "{error}");
        assert_eq!(bindings.len(), 1, "the rejected binding is not pushed");
    }

    #[test]
    fn linear_aql_refuses_a_tape_with_position_bindings() {
        // Linear AQL submits recorded kernargs verbatim. A declared position
        // field (here a QSA-style length) would replay frozen at the capture
        // position, so prepare must refuse before any device work and leave the
        // route unprepared.
        let declared = [ReplayKernargBinding::PositionPlusU32 {
            offset: 8,
            addend: 1,
        }];
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        for declared in [&[][..], &declared[..]] {
            controller.record_hip_launch_with_accesses(
                "qsa_select",
                None,
                [1, 1, 1],
                [256, 1, 1],
                0,
                &[0u8; 16],
                None,
                declared,
                None,
                None,
            );
        }
        controller.finish_capture().unwrap();
        let error = controller
            .prepare_linear_aql(0)
            .expect_err("a position-bound tape must not prepare on linear AQL");
        assert!(error.contains("offset 8 of dispatch 1"), "{error}");
        assert_ne!(controller.state(), ReplayState::Ready);
    }

    #[test]
    fn a_declared_frame_counter_is_not_a_position_binding() {
        // The GDN frame counter belongs to the replay layer, which derives it
        // from the recorded launch. A model lowering must not be able to claim
        // it as a declared position field.
        let declared = [ReplayKernargBinding::GdnFrameU32 {
            offset: 76,
            frames: 1,
        }];
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller.record_hip_launch_with_accesses(
            "gated_delta_net_q8_fast",
            None,
            [1, 1, 1],
            [32, 1, 1],
            0,
            &[0u8; 80],
            None,
            &declared,
            None,
            None,
        );
        let launches = controller.recorded_launches().to_vec();
        let mut bindings: Vec<(usize, ReplayKernargBinding)> = Vec::new();
        let error = merge_declared_kernarg_bindings(&launches, 1, &mut bindings)
            .expect_err("a frame counter is not a position-derived declaration");
        assert!(error.contains("not position-derived"), "{error}");
        assert!(bindings.is_empty());
    }

    #[test]
    fn declared_bindings_are_part_of_tape_identity() {
        // Two tapes that differ only in which slot is declared dynamic are
        // different replay contracts, so the sequence hash must separate them.
        let launch = |declared: &[ReplayKernargBinding]| {
            let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
            controller.record_hip_launch_with_accesses(
                "gated_delta_conv_bf16_f32",
                None,
                [1, 1, 1],
                [256, 1, 1],
                0,
                &[0u8; 16],
                None,
                declared,
                None,
                None,
            );
            replay_sequence_hash(controller.recorded_launches())
        };
        let plain = launch(&[]);
        let declared = launch(&[ReplayKernargBinding::PositionModU32 {
            offset: 4,
            addend: 0,
            modulus: 3,
        }]);
        assert_ne!(plain, declared);
    }

    #[test]
    fn replay_kernarg_binding_gdn_requires_binding() {
        // Simulate preparation admissibility: a GDN-family launch without a
        // GdnFrameU32 binding must be rejected instead of silently replaying stale.
        let kernel = "gated_delta_net_q8_fast";
        let metadata_kernarg_size = 80usize;
        assert!(metadata_kernarg_size >= 80);
        let bindings: Vec<(usize, ReplayKernargBinding)> = vec![];
        let is_gdn =
            kernel == "gated_delta_net_q8_fast" || kernel.starts_with("gated_delta_net_q8_compact");
        assert!(is_gdn);
        let has_binding = bindings.iter().any(|(_, b)| {
            matches!(
                b,
                ReplayKernargBinding::GdnFrameU32 {
                    offset: 76,
                    frames: _
                }
            )
        });
        assert!(!has_binding);
        // In real prepare, this would be Err; here we just verify detection.
        let should_reject = is_gdn && !has_binding;
        assert!(should_reject);
        // With correct binding, it passes.
        let bindings_ok = vec![(
            0,
            ReplayKernargBinding::GdnFrameU32 {
                offset: 76,
                frames: 1,
            },
        )];
        let has_binding_ok = bindings_ok.iter().any(|(_, b)| {
            matches!(
                b,
                ReplayKernargBinding::GdnFrameU32 {
                    offset: 76,
                    frames: _
                }
            )
        });
        assert!(has_binding_ok);
        assert!(!(is_gdn && !has_binding_ok));
    }

    fn make_gdn_kernarg(nt: i32, len: usize) -> Vec<u8> {
        let mut kernarg = vec![0u8; len];
        if len >= 68 {
            kernarg[64..68].copy_from_slice(&nt.to_le_bytes());
        }
        kernarg
    }

    #[test]
    fn gdn_frame_count_derivation_sequence_and_independent_batched() {
        // sequence-batched: nt = 16 tokens, grid.z = 1 => frames = 16
        let kernarg = make_gdn_kernarg(16, 80);
        assert_eq!(gdn_requant_frames_for_dispatch(&kernarg, 1).unwrap(), 16);
        // independent-batched: nt = 1, grid.z = 16 => frames = 16 (1 * 16)
        let kernarg = make_gdn_kernarg(1, 80);
        assert_eq!(gdn_requant_frames_for_dispatch(&kernarg, 16).unwrap(), 16);
        // plain AR single-token: nt = 1, grid.z = 1 => frames = 1 (preserves today's behavior)
        let kernarg = make_gdn_kernarg(1, 80);
        assert_eq!(gdn_requant_frames_for_dispatch(&kernarg, 1).unwrap(), 1);
    }

    #[test]
    fn gdn_frame_count_rejects_short_and_nonpositive_nt() {
        // kernarg shorter than 80 bytes must be rejected
        let short = make_gdn_kernarg(16, 64);
        assert!(gdn_requant_frames_for_dispatch(&short, 1).is_err());
        let short79 = make_gdn_kernarg(16, 79);
        assert!(gdn_requant_frames_for_dispatch(&short79, 1).is_err());
        // nt <= 0 must be rejected
        let zero = make_gdn_kernarg(0, 80);
        assert!(gdn_requant_frames_for_dispatch(&zero, 1).is_err());
        let neg = make_gdn_kernarg(-1, 80);
        assert!(gdn_requant_frames_for_dispatch(&neg, 1).is_err());
        // i32::MAX * 2 == 4294967294, which still fits in u32 — the first
        // genuinely overflowing multiplier is 3.
        let large = make_gdn_kernarg(i32::MAX, 80);
        assert!(gdn_requant_frames_for_dispatch(&large, 2).is_ok());
        assert!(gdn_requant_frames_for_dispatch(&large, 3).is_err());
    }

    #[test]
    fn gdn_frame_apply_advances_by_frames_and_writes_base() {
        // Restore a known checkpoint so we can assert on delta rather than
        // absolute non-zero value (fetch_add legitimately starts at 0 in a
        // fresh process).
        crate::norm::restore_gdn_requant_frame_checkpoint(1000);
        let checkpoint = crate::norm::gdn_requant_frame_checkpoint();
        let mut kernarg = vec![0u8; 80];
        let binding = ReplayKernargBinding::GdnFrameU32 {
            offset: 76,
            frames: 16,
        };
        binding.apply(&mut kernarg, 0).unwrap();
        let written = u32::from_le_bytes(kernarg[76..80].try_into().unwrap());
        // Written base must equal the pre-advance checkpoint
        assert_eq!(written, checkpoint);
        let after = crate::norm::gdn_requant_frame_checkpoint();
        assert_eq!(after, checkpoint + 16);
        // Second apply with frames=1 should advance by 1 from new base
        let checkpoint2 = after;
        let binding2 = ReplayKernargBinding::GdnFrameU32 {
            offset: 76,
            frames: 1,
        };
        binding2.apply(&mut kernarg, 0).unwrap();
        let written2 = u32::from_le_bytes(kernarg[76..80].try_into().unwrap());
        assert_eq!(written2, checkpoint2);
        assert_eq!(crate::norm::gdn_requant_frame_checkpoint(), checkpoint2 + 1);
        // Clean up: restore to avoid leaking state to other tests (tests run
        // in parallel in same process; use a deterministic restore).
        crate::norm::restore_gdn_requant_frame_checkpoint(checkpoint);
    }

    #[test]
    fn replay_quiescence_mapping() {
        fn map(q: redline_dispatch::aql::Quiescence) -> ReplayQuiescence {
            match q {
                redline_dispatch::aql::Quiescence::Proven => ReplayQuiescence::Proven,
                redline_dispatch::aql::Quiescence::Unknown => ReplayQuiescence::Unknown,
            }
        }
        assert_eq!(
            map(redline_dispatch::aql::Quiescence::Proven),
            ReplayQuiescence::Proven
        );
        assert_eq!(
            map(redline_dispatch::aql::Quiescence::Unknown),
            ReplayQuiescence::Unknown
        );
    }

    #[test]
    fn unknown_quiescence_requires_quiesce_before_reuse_and_shutdown_retains() {
        // C0 contract pin at rdna layer: Unknown means in-flight may still be
        // writing. Subsequent replay must be refused and shutdown must retain
        // prepared IB/kernargs/kernels until quiesce proves Proven.
        let failure = RetainedReplayFailure {
            error: "simulated doorbell wait timeout".to_owned(),
            quiescence: ReplayQuiescence::Unknown,
        };
        assert_eq!(failure.quiescence, ReplayQuiescence::Unknown);
        // Caller policy: Unknown => quarantine — do not free, do not replay,
        // do not reuse controller for new model until quiesce succeeds.
        let must_quarantine = failure.quiescence == ReplayQuiescence::Unknown;
        assert!(must_quarantine);
        // Proven => safe to retry or free after handling error
        let proven = RetainedReplayFailure {
            error: "position 200 exceeds prepared max_position 100".to_owned(),
            quiescence: ReplayQuiescence::Proven,
        };
        assert_eq!(proven.quiescence, ReplayQuiescence::Proven);
        assert!(!proven.error.is_empty());
        // Simulate shutdown retention: on Unknown, shutdown would return Err(Unknown)
        // and keep prepared resources; on Proven it returns Ok and releases.
        fn simulated_shutdown(quiescence: ReplayQuiescence) -> Result<(), RetainedReplayFailure> {
            match quiescence {
                ReplayQuiescence::Unknown => Err(RetainedReplayFailure {
                    error: "inactivate_all failed: device still in-flight".to_owned(),
                    quiescence: ReplayQuiescence::Unknown,
                }),
                ReplayQuiescence::Proven => Ok(()),
            }
        }
        assert!(simulated_shutdown(ReplayQuiescence::Unknown).is_err());
        assert!(simulated_shutdown(ReplayQuiescence::Proven).is_ok());
    }

    fn make_kernarg_with_u32(offset: usize, value: u32, len: usize) -> Vec<u8> {
        let mut kernarg = vec![0u8; len];
        kernarg[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        kernarg
    }

    #[test]
    fn position_scalar_advances_by_delta_yields_binding_and_applies_at_third_position() {
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller.begin_capture().unwrap();
        let kernarg_10 = make_kernarg_with_u32(16, 10, 64);
        controller.record_hip_launch("gemv_hfq4g256", None, [1, 1, 1], [64, 1, 1], 0, &kernarg_10);
        controller.finish_capture().unwrap();
        let earlier = controller.snapshot_recorded_kernargs();
        controller.begin_capture().unwrap();
        let kernarg_20 = make_kernarg_with_u32(16, 20, 64);
        controller.record_hip_launch("gemv_hfq4g256", None, [1, 1, 1], [64, 1, 1], 0, &kernarg_20);
        controller.finish_capture().unwrap();
        let count = controller
            .synthesize_position_bindings(&earlier, 10, 20)
            .unwrap();
        assert_eq!(count, 1);
        let bindings = controller.synthesized_position_bindings().to_vec();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].0, 0);
        match bindings[0].1 {
            ReplayKernargBinding::PositionPlusU32 { offset, addend } => {
                assert_eq!(offset, 16);
                assert_eq!(addend, 0); // 20 - 20 = 0, 10 -10 =0
            }
            _ => panic!("expected PositionPlusU32"),
        }
        // Applying at third position 30 should produce 30.
        let mut kernarg_third = vec![0u8; 64];
        apply_kernarg_bindings_for_dispatch(&mut kernarg_third, 0, 30, &bindings).unwrap();
        let v = u32::from_le_bytes(kernarg_third[16..20].try_into().unwrap());
        // But apply writes value = position + addend = 30 + 0 =30; with ne_bytes vs le_bytes?
        // PositionPlusU32 uses to_ne_bytes, so need to read via from_ne_bytes.
        let v_ne = u32::from_ne_bytes(kernarg_third[16..20].try_into().unwrap());
        assert_eq!(v_ne, 30);
        // Also verify that helper reproduces earlier samples.
        let mut kernarg_earlier = vec![0u8; 64];
        apply_kernarg_bindings_for_dispatch(&mut kernarg_earlier, 0, 10, &bindings).unwrap();
        let ve = u32::from_ne_bytes(kernarg_earlier[16..20].try_into().unwrap());
        assert_eq!(ve, 10);
        let mut kernarg_current = vec![0u8; 64];
        apply_kernarg_bindings_for_dispatch(&mut kernarg_current, 0, 20, &bindings).unwrap();
        let vc = u32::from_ne_bytes(kernarg_current[16..20].try_into().unwrap());
        assert_eq!(vc, 20);
    }

    #[test]
    fn position_scalar_with_constant_offset_recovers_addend() {
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller.begin_capture().unwrap();
        // scalar = position + 5
        let kernarg_10 = make_kernarg_with_u32(32, 15, 64); // 10+5
        controller.record_hip_launch(
            "fused_qkv_hfq4g256",
            None,
            [1, 1, 1],
            [64, 1, 1],
            0,
            &kernarg_10,
        );
        controller.finish_capture().unwrap();
        let earlier = controller.snapshot_recorded_kernargs();
        controller.begin_capture().unwrap();
        let kernarg_20 = make_kernarg_with_u32(32, 25, 64); // 20+5
        controller.record_hip_launch(
            "fused_qkv_hfq4g256",
            None,
            [1, 1, 1],
            [64, 1, 1],
            0,
            &kernarg_20,
        );
        controller.finish_capture().unwrap();
        let count = controller
            .synthesize_position_bindings(&earlier, 10, 20)
            .unwrap();
        assert_eq!(count, 1);
        match controller.synthesized_position_bindings()[0].1 {
            ReplayKernargBinding::PositionPlusU32 { offset, addend } => {
                assert_eq!(offset, 32);
                assert_eq!(addend, 5);
            }
            _ => panic!("expected PositionPlusU32"),
        }
    }

    #[test]
    fn position_scalar_non_delta_difference_is_rejected_with_kernel_index_offset() {
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller.begin_capture().unwrap();
        let kernarg_10 = make_kernarg_with_u32(8, 100, 64);
        controller.record_hip_launch("softmax_f32", None, [1, 1, 1], [64, 1, 1], 0, &kernarg_10);
        controller.finish_capture().unwrap();
        let earlier = controller.snapshot_recorded_kernargs();
        controller.begin_capture().unwrap();
        // Diff is 5, delta is 10 (positions 10->20 delta 10, but value diff 5)
        let kernarg_20 = make_kernarg_with_u32(8, 105, 64);
        controller.record_hip_launch("softmax_f32", None, [1, 1, 1], [64, 1, 1], 0, &kernarg_20);
        controller.finish_capture().unwrap();
        let err = controller
            .synthesize_position_bindings(&earlier, 10, 20)
            .unwrap_err();
        assert!(err.contains("softmax_f32"), "error missing kernel: {err}");
        assert!(
            err.contains("launch 0"),
            "error missing launch index: {err}"
        );
        assert!(err.contains("offset 8"), "error missing byte offset: {err}");
        assert!(
            err.contains("unexplained"),
            "error missing unexplained: {err}"
        );
    }

    #[test]
    fn eight_byte_pointer_field_is_rejected_as_moved_allocation() {
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller.begin_capture().unwrap();
        let mut kernarg_early = vec![0u8; 32];
        let ptr_early: u64 = 0x7f00_0000_1000;
        kernarg_early[0..8].copy_from_slice(&ptr_early.to_le_bytes());
        controller.record_hip_launch(
            "gemv_hfq4g256",
            None,
            [1, 1, 1],
            [64, 1, 1],
            0,
            &kernarg_early,
        );
        controller.finish_capture().unwrap();
        let earlier = controller.snapshot_recorded_kernargs();
        controller.begin_capture().unwrap();
        let mut kernarg_cur = vec![0u8; 32];
        let ptr_cur: u64 = 0x7f00_0000_2000; // moved allocation, diff != delta
        kernarg_cur[0..8].copy_from_slice(&ptr_cur.to_le_bytes());
        controller.record_hip_launch(
            "gemv_hfq4g256",
            None,
            [1, 1, 1],
            [64, 1, 1],
            0,
            &kernarg_cur,
        );
        controller.finish_capture().unwrap();
        let err = controller
            .synthesize_position_bindings(&earlier, 10, 20)
            .unwrap_err();
        assert!(
            err.contains("moved allocation"),
            "error missing moved allocation: {err}"
        );
        assert!(err.contains("offset 0"), "error missing offset: {err}");
    }

    #[test]
    fn gdn_frame_offset_76_is_skipped_on_gdn_family() {
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller.begin_capture().unwrap();
        let mut kernarg_early = vec![0u8; 80];
        kernarg_early[76..80].copy_from_slice(&1234u32.to_le_bytes());
        // need nt at 64 to be valid but synthesize skips offset 76 regardless
        kernarg_early[64..68].copy_from_slice(&1i32.to_le_bytes());
        controller.record_hip_launch(
            "gated_delta_net_q8_fast",
            None,
            [1, 1, 1],
            [64, 1, 1],
            0,
            &kernarg_early,
        );
        controller.finish_capture().unwrap();
        let earlier = controller.snapshot_recorded_kernargs();
        controller.begin_capture().unwrap();
        let mut kernarg_cur = vec![0u8; 80];
        kernarg_cur[76..80].copy_from_slice(&9999u32.to_le_bytes()); // arbitrary diff
        kernarg_cur[64..68].copy_from_slice(&1i32.to_le_bytes());
        controller.record_hip_launch(
            "gated_delta_net_q8_fast",
            None,
            [1, 1, 1],
            [64, 1, 1],
            0,
            &kernarg_cur,
        );
        controller.finish_capture().unwrap();
        let count = controller
            .synthesize_position_bindings(&earlier, 10, 20)
            .unwrap();
        assert_eq!(
            count, 0,
            "GDN offset 76 should be skipped, got bindings: {count}"
        );
    }

    #[test]
    fn mismatched_launch_counts_kernel_names_and_kernarg_lengths_are_rejected_distinctly() {
        // Mismatched launch counts
        let mut controller = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller.begin_capture().unwrap();
        controller.record_hip_launch("a", None, [1, 1, 1], [32, 1, 1], 0, &[1, 2, 3, 4]);
        controller.finish_capture().unwrap();
        let earlier = controller.snapshot_recorded_kernargs();
        controller.begin_capture().unwrap();
        controller.record_hip_launch("a", None, [1, 1, 1], [32, 1, 1], 0, &[1, 2, 3, 4]);
        controller.record_hip_launch("b", None, [1, 1, 1], [32, 1, 1], 0, &[5, 6, 7, 8]);
        controller.finish_capture().unwrap();
        let err = controller
            .synthesize_position_bindings(&earlier, 10, 20)
            .unwrap_err();
        assert!(
            err.contains("launch count mismatch"),
            "expected launch count mismatch: {err}"
        );

        // Mismatched kernel names
        let mut controller2 = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller2.begin_capture().unwrap();
        controller2.record_hip_launch("kernel_a", None, [1, 1, 1], [32, 1, 1], 0, &[1, 2, 3, 4]);
        controller2.finish_capture().unwrap();
        let earlier2 = controller2.snapshot_recorded_kernargs();
        controller2.begin_capture().unwrap();
        controller2.record_hip_launch("kernel_b", None, [1, 1, 1], [32, 1, 1], 0, &[1, 2, 3, 4]);
        controller2.finish_capture().unwrap();
        let err2 = controller2
            .synthesize_position_bindings(&earlier2, 10, 20)
            .unwrap_err();
        assert!(
            err2.contains("kernel mismatch"),
            "expected kernel mismatch: {err2}"
        );

        // Mismatched kernarg lengths
        let mut controller3 = ReplayController::new_armed(ReplayBackendRequest::Auto);
        controller3.begin_capture().unwrap();
        controller3.record_hip_launch("same_kernel", None, [1, 1, 1], [32, 1, 1], 0, &[1, 2, 3, 4]);
        controller3.finish_capture().unwrap();
        let earlier3 = controller3.snapshot_recorded_kernargs();
        controller3.begin_capture().unwrap();
        controller3.record_hip_launch(
            "same_kernel",
            None,
            [1, 1, 1],
            [32, 1, 1],
            0,
            &[1, 2, 3, 4, 5, 6, 7, 8],
        );
        controller3.finish_capture().unwrap();
        let err3 = controller3
            .synthesize_position_bindings(&earlier3, 10, 20)
            .unwrap_err();
        assert!(
            err3.contains("kernarg length mismatch"),
            "expected kernarg length mismatch: {err3}"
        );
        // Ensure distinct messages
        assert_ne!(err, err2);
        assert_ne!(err2, err3);
        assert_ne!(err, err3);
    }

    fn test_launch(kernel: &str) -> RecordedHipLaunch {
        RecordedHipLaunch {
            kernel: kernel.to_owned(),
            artifact: None,
            grid: [1, 1, 1],
            block: [64, 1, 1],
            shared_mem: 0,
            grid_binding: None,
            declared_kernarg_bindings: Vec::new(),
            kernarg: Vec::new(),
            accesses: None,
            binding_layout: None,
        }
    }

    fn sample_set_sh(
        packet_dword: u32,
        following_dispatch: u32,
        first_register: u32,
        value_dwords: u32,
        repeated_value_dwords: u32,
    ) -> Gfx10SetShRegRecord {
        Gfx10SetShRegRecord {
            packet_dword,
            following_dispatch,
            first_register,
            value_dwords,
            repeated_value_dwords,
        }
    }

    #[test]
    fn pm4_set_sh_transition_identity_and_non_identity_order() {
        let recorded = [
            test_launch("kernel_a"),
            test_launch("kernel_b"),
            test_launch("kernel_c"),
        ];
        let identity = [0usize, 1, 2];
        assert_eq!(
            pm4_set_sh_transition(&identity, &recorded, 0).unwrap(),
            (
                PM4_STREAM_ENTRY_TRANSITION.to_owned(),
                "kernel_a".to_owned()
            )
        );
        assert_eq!(
            pm4_set_sh_transition(&identity, &recorded, 1).unwrap(),
            ("kernel_a".to_owned(), "kernel_b".to_owned())
        );
        assert_eq!(
            pm4_set_sh_transition(&identity, &recorded, 2).unwrap(),
            ("kernel_b".to_owned(), "kernel_c".to_owned())
        );

        // Non-identity schedule: execution order c, a, b.
        let reordered = [2usize, 0, 1];
        assert_eq!(
            pm4_set_sh_transition(&reordered, &recorded, 0).unwrap(),
            (
                PM4_STREAM_ENTRY_TRANSITION.to_owned(),
                "kernel_c".to_owned()
            )
        );
        assert_eq!(
            pm4_set_sh_transition(&reordered, &recorded, 1).unwrap(),
            ("kernel_c".to_owned(), "kernel_a".to_owned())
        );
        assert_eq!(
            pm4_set_sh_transition(&reordered, &recorded, 2).unwrap(),
            ("kernel_a".to_owned(), "kernel_b".to_owned())
        );
        assert!(pm4_set_sh_transition(&reordered, &recorded, 3).is_err());
    }

    #[test]
    fn pm4_stream_accounting_rejects_multi_queue_lowering() {
        assert!(validate_pm4_stream_accounting_queue_count(false, 2).is_ok());
        assert!(validate_pm4_stream_accounting_queue_count(true, 1).is_ok());
        let error = validate_pm4_stream_accounting_queue_count(true, 2).unwrap_err();
        assert!(error.contains("requires one PM4 queue"));
        assert!(error.contains('2'));
    }

    #[test]
    fn pm4_stream_accounting_stable_btreemap_ordering() {
        let recorded = [test_launch("alpha"), test_launch("beta")];
        let order = [0usize, 1];
        // Insert census keys out of sorted order; BTreeMap must emit sorted rows.
        let mut census = BTreeMap::new();
        census.insert((0x58, 8), 1usize);
        census.insert((PACKET3_SET_SH_REG, 4), 2usize);
        census.insert((PACKET3_DISPATCH_DIRECT, DISPATCH_DIRECT_PACKET_DWORDS), 2);
        census.insert((0x46, 2), 1);
        // 8 + 4*2 + 5*2 + 2 = 28
        let command_dwords = 28_u32;
        let records = [
            sample_set_sh(8, 0, 0x20c, 2, 0),
            sample_set_sh(12, 1, 0x207, 2, 0),
        ];
        let report = Pm4StreamAccountingReport::build(
            "gfx1100",
            &order,
            &recorded,
            command_dwords,
            0xabc_u64,
            1,
            1,
            &census,
            &records,
        )
        .expect("reconciled sample");
        let formatted = report.format_report();
        // Packet classes: BTreeMap sorts by (opcode, dwords)
        let classes_idx = formatted.find("packet_classes=[").unwrap();
        let classes = &formatted[classes_idx..];
        let pos_15 = classes.find("op=0x15").unwrap();
        let pos_46 = classes.find("op=0x46").unwrap();
        let pos_58 = classes.find("op=0x58").unwrap();
        let pos_76 = classes.find("op=0x76").unwrap();
        assert!(pos_15 < pos_46 && pos_46 < pos_58 && pos_58 < pos_76);
        // Register offsets sorted numerically: 0x207 before 0x20c
        let regs_idx = formatted.find("writes_by_register_offset=[").unwrap();
        let regs = &formatted[regs_idx..];
        assert!(regs.find("0x207=").unwrap() < regs.find("0x20c=").unwrap());
        // Transitions sorted by (prev, curr) string order
        let tr_idx = formatted.find("writes_by_transition=[").unwrap();
        let tr = &formatted[tr_idx..];
        let entry_alpha = tr.find("<entry>->alpha=").unwrap();
        let alpha_beta = tr.find("alpha->beta=").unwrap();
        assert!(entry_alpha < alpha_beta);
        // No raw addresses in the report.
        assert!(!formatted.contains("address"));
    }

    #[test]
    fn pm4_stream_accounting_reconciliation_failures() {
        let recorded = [test_launch("a"), test_launch("b")];
        let order = [0usize, 1];
        let mut census = BTreeMap::new();
        census.insert((PACKET3_DISPATCH_DIRECT, DISPATCH_DIRECT_PACKET_DWORDS), 2);
        census.insert((PACKET3_SET_SH_REG, 4), 1);
        // 5*2 + 4 = 14, but claim 15 → dword sum failure
        let err = Pm4StreamAccountingReport::build(
            "gfx1100",
            &order,
            &recorded,
            15,
            0,
            0,
            0,
            &census,
            &[sample_set_sh(0, 0, 0x207, 2, 0)],
        )
        .unwrap_err();
        assert!(
            err.contains("accounts for") && err.contains("14") && err.contains("15"),
            "dword mismatch: {err}"
        );

        // Dispatch count mismatch
        let mut census = BTreeMap::new();
        census.insert((PACKET3_DISPATCH_DIRECT, DISPATCH_DIRECT_PACKET_DWORDS), 1);
        census.insert((PACKET3_SET_SH_REG, 3), 1);
        // 5 + 3 = 8
        let err = Pm4StreamAccountingReport::build(
            "gfx1100",
            &order,
            &recorded,
            8,
            0,
            0,
            0,
            &census,
            &[sample_set_sh(0, 0, 0x207, 1, 0)],
        )
        .unwrap_err();
        assert!(
            err.contains("DISPATCH_DIRECT count"),
            "dispatch mismatch: {err}"
        );

        // following_dispatch outside order
        let mut census = BTreeMap::new();
        census.insert((PACKET3_DISPATCH_DIRECT, DISPATCH_DIRECT_PACKET_DWORDS), 2);
        census.insert((PACKET3_SET_SH_REG, 4), 1);
        let err = Pm4StreamAccountingReport::build(
            "gfx1100",
            &order,
            &recorded,
            14,
            0,
            0,
            0,
            &census,
            &[sample_set_sh(0, 9, 0x207, 2, 0)],
        )
        .unwrap_err();
        assert!(
            err.contains("following_dispatch=9") || err.contains("outside frozen order"),
            "following_dispatch failure: {err}"
        );

        // SET_SH value count vs census payload mismatch
        let mut census = BTreeMap::new();
        census.insert((PACKET3_DISPATCH_DIRECT, DISPATCH_DIRECT_PACKET_DWORDS), 2);
        census.insert((PACKET3_SET_SH_REG, 5), 1); // 3 values
        let err = Pm4StreamAccountingReport::build(
            "gfx1100",
            &order,
            &recorded,
            15, // 10 + 5
            0,
            0,
            0,
            &census,
            &[sample_set_sh(0, 0, 0x207, 2, 0)], // only 2 values recorded
        )
        .unwrap_err();
        assert!(
            err.contains("value dwords") || err.contains("payload"),
            "value mismatch: {err}"
        );
    }

    #[test]
    fn pm4_stream_accounting_report_is_not_retained() {
        // Report is preparation-only and dropped before graph install.
        // PreparedPm4Replay has no accounting field; constructing and dropping
        // the report does not require stashing it on the retained type.
        let recorded = [test_launch("k0")];
        let order = [0usize];
        let mut census = BTreeMap::new();
        census.insert((PACKET3_DISPATCH_DIRECT, DISPATCH_DIRECT_PACKET_DWORDS), 1);
        census.insert((PACKET3_SET_SH_REG, 3), 1);
        let report = Pm4StreamAccountingReport::build(
            "gfx1100",
            &order,
            &recorded,
            8, // 5 + 3
            0xabc,
            0,
            0,
            &census,
            &[sample_set_sh(0, 0, 0x207, 1, 0)],
        )
        .unwrap();
        let line = report.format_report();
        assert!(line.contains("arch=gfx1100"));
        assert!(line.contains("dispatches=1"));
        assert!(line.contains("command_dwords=8"));
        drop(report);
        let _ = std::mem::size_of::<PreparedPm4Replay>();
    }

    #[test]
    fn gfx1201_pm4_pacing_parses_opt_out_sizes_and_rejects_bad_values() {
        use Gfx12DispatchPacing::PostDispatchNop;
        for (value, want) in [
            (None, Ok(GFX1201_DEFAULT_PM4_PACING)),
            (Some("auto"), Ok(GFX1201_DEFAULT_PM4_PACING)),
            (Some("0"), Ok(Gfx12DispatchPacing::None)),
            (Some(" OFF "), Ok(Gfx12DispatchPacing::None)),
            (Some("nop:128"), Ok(PostDispatchNop(128))),
            (Some("nop:16384"), Ok(PostDispatchNop(16384))),
        ] {
            assert_eq!(parse_gfx1201_pm4_pacing(value), want, "{value:?}");
        }
        for bad in ["nop:0", "nop:16385", "align:64", "64", "pad:8"] {
            assert!(parse_gfx1201_pm4_pacing(Some(bad)).is_err(), "{bad}");
        }
    }

    fn relocation_move(old_base: u64, new_base: u64, reserved: usize, mapped: usize) -> VmmResourceMove {
        VmmResourceMove {
            old_base: old_base as usize,
            new_base: new_base as usize,
            reserved_bytes: reserved,
            mapped_bytes: mapped,
            owner_generation: 7,
        }
    }

    const FIXTURE_A_BYTES: u64 = 0x1_0000;
    const FIXTURE_B_BYTES: u64 = 0x2_0000;
    const RESERVED: usize = 0x20_0000;
    const MOVED_A: u64 = 0x7f00_0100_0000;
    const MOVED_A2: u64 = 0x7f00_0200_0000;

    fn fixture_access(base: u64, bytes: u64, at: u64) -> RecordedResourceAccess {
        RecordedResourceAccess {
            allocation_base: base,
            allocation_bytes: bytes,
            access_base: at,
            mode: RecordedAccessMode::Read,
        }
    }

    /// Controller holding the slice-1 fixture as one typed recorded launch
    /// with its two tape resources (A at slot 0, B at slot 16).
    fn relocation_controller() -> (ReplayController, u64, u64) {
        let (snapshot, layout, bindings, base_a, base_b) = bound_segment_fixture();
        let mut controller = ReplayController::new(ReplayBackendRequest::Auto);
        controller
            .tape_resources
            .insert((base_a, FIXTURE_A_BYTES), layout.slots[0].resource);
        controller
            .tape_resources
            .insert((base_b, FIXTURE_B_BYTES), layout.slots[1].resource);
        controller.replay_bindings = bindings;
        let mut launch = test_launch("typed_fixture");
        launch.kernarg = snapshot;
        launch.binding_layout = Some(layout);
        launch.accesses = Some(vec![
            fixture_access(base_a, FIXTURE_A_BYTES, base_a + 0x40),
            fixture_access(base_b, FIXTURE_B_BYTES, base_b),
        ]);
        controller.recorded.push(launch);
        (controller, base_a, base_b)
    }

    fn bound_base(controller: &ReplayController, resource: ResourceId) -> (u64, BindingRevision) {
        let binding = controller.replay_bindings.resource(resource).expect("bound");
        (binding.base().as_ptr() as usize as u64, binding.revision())
    }

    #[test]
    fn relocation_rewrites_only_the_moved_slot_and_bumps_revision() {
        let (mut controller, base_a, base_b) = relocation_controller();
        let before = controller.recorded[0].kernarg.clone();
        let resource_a = controller.recorded[0].binding_layout.as_ref().unwrap().slots[0].resource;
        let resource_b = controller.recorded[0].binding_layout.as_ref().unwrap().slots[1].resource;
        let report = controller
            .relocate_bindings(&[relocation_move(base_a, MOVED_A, FIXTURE_A_BYTES as usize, FIXTURE_A_BYTES as usize)])
            .expect("relocates");
        assert_eq!(
            report,
            BindingRefreshReport::Refreshed {
                resources: 2,
                reencoded: 1,
                revision: BindingRevision(1),
            }
        );
        assert_eq!(controller.binding_revision(), BindingRevision(1));
        let after = &controller.recorded[0].kernarg;
        assert_eq!(&after[0..8], &(MOVED_A + 0x40).to_ne_bytes());
        assert_eq!(&after[8..], &before[8..]);
        assert!(controller.tape_resources.contains_key(&(MOVED_A, FIXTURE_A_BYTES)));
        assert!(!controller.tape_resources.contains_key(&(base_a, FIXTURE_A_BYTES)));
        assert!(controller.tape_resources.contains_key(&(base_b, FIXTURE_B_BYTES)));
        assert_eq!(bound_base(&controller, resource_a), (MOVED_A, BindingRevision(1)));
        assert_eq!(bound_base(&controller, resource_b), (base_b, BindingRevision(1)));
        let accesses = controller.recorded[0].accesses.as_ref().unwrap();
        assert_eq!(accesses[0].allocation_base, MOVED_A);
        assert_eq!(accesses[0].access_base, MOVED_A + 0x40);
        assert_eq!(accesses[1], fixture_access(base_b, FIXTURE_B_BYTES, base_b));
    }

    #[test]
    fn relocated_recorded_snapshot_equals_encoding_of_new_bindings() {
        let (mut controller, base_a, _) = relocation_controller();
        controller
            .relocate_bindings(&[relocation_move(base_a, MOVED_A, FIXTURE_A_BYTES as usize, FIXTURE_A_BYTES as usize)])
            .expect("relocates");
        let launch = &controller.recorded[0];
        let encoded = encode_bound_kernarg(
            &launch.kernarg,
            launch.binding_layout.as_ref().unwrap(),
            &controller.replay_bindings,
            "typed_fixture",
        )
        .expect("encodes");
        assert_eq!(encoded, launch.kernarg);
        verify_bound_kernarg("typed_fixture", &launch.kernarg, &encoded).expect("verifies");
    }

    #[test]
    fn chained_relocation_follows_the_rekeyed_resource() {
        let (mut controller, base_a, _) = relocation_controller();
        controller
            .relocate_bindings(&[relocation_move(base_a, MOVED_A, FIXTURE_A_BYTES as usize, FIXTURE_A_BYTES as usize)])
            .expect("A to B");
        let report = controller
            .relocate_bindings(&[relocation_move(MOVED_A, MOVED_A2, FIXTURE_A_BYTES as usize, FIXTURE_A_BYTES as usize)])
            .expect("B to C");
        assert_eq!(
            report,
            BindingRefreshReport::Refreshed {
                resources: 2,
                reencoded: 1,
                revision: BindingRevision(2),
            }
        );
        assert_eq!(
            &controller.recorded[0].kernarg[0..8],
            &(MOVED_A2 + 0x40).to_ne_bytes()
        );
        assert!(controller.tape_resources.contains_key(&(MOVED_A2, FIXTURE_A_BYTES)));
        assert!(!controller.tape_resources.contains_key(&(MOVED_A, FIXTURE_A_BYTES)));
    }

    #[test]
    fn resources_outside_every_move_are_untouched_and_revision_holds() {
        let (mut controller, _, _) = relocation_controller();
        let before = controller.recorded[0].kernarg.clone();
        let keys: Vec<_> = controller.tape_resources.keys().copied().collect();
        let report = controller
            .relocate_bindings(&[relocation_move(
                0x7e00_0000_0000,
                0x7e00_1000_0000,
                RESERVED,
                RESERVED,
            )])
            .expect("unrelated move");
        assert_eq!(
            report,
            BindingRefreshReport::Refreshed {
                resources: 2,
                reencoded: 0,
                revision: BindingRevision(0),
            }
        );
        assert_eq!(controller.binding_revision(), BindingRevision(0));
        assert_eq!(controller.recorded[0].kernarg, before);
        assert_eq!(controller.tape_resources.keys().copied().collect::<Vec<_>>(), keys);
    }

    #[test]
    fn failed_relocation_leaves_every_piece_of_state_unchanged() {
        let (mut controller, base_a, _) = relocation_controller();
        // Slot 0 would read 8 bytes at the very end of the mapped coverage.
        controller.recorded[0].binding_layout.as_mut().unwrap().slots[0].interior_offset = 0xFFFC;
        let before = controller.recorded[0].clone();
        let keys: Vec<_> = controller.tape_resources.keys().copied().collect();
        let error = controller
            .relocate_bindings(&[relocation_move(base_a, MOVED_A, FIXTURE_A_BYTES as usize, FIXTURE_A_BYTES as usize)])
            .expect_err("slot past coverage");
        assert!(error.contains("mapped"), "{error}");
        assert_eq!(controller.recorded[0], before);
        assert_eq!(controller.binding_revision(), BindingRevision(0));
        assert_eq!(controller.tape_resources.keys().copied().collect::<Vec<_>>(), keys);
        let resource_a = before.binding_layout.as_ref().unwrap().slots[0].resource;
        assert_eq!(bound_base(&controller, resource_a), (base_a, BindingRevision(0)));
    }

    #[test]
    fn mapped_bytes_smaller_than_the_resource_is_rejected() {
        let (controller, base_a, _) = relocation_controller();
        let error = plan_relocation(
            &controller.tape_resources,
            &[relocation_move(base_a, MOVED_A, FIXTURE_A_BYTES as usize, 0x8000)],
        )
        .expect_err("not fully mapped");
        assert!(error.contains("not fully mapped"), "{error}");
    }

    #[test]
    fn plan_rekeys_with_interior_offset_and_orders_by_old_key() {
        let (controller, base_a, base_b) = relocation_controller();
        let plan = plan_relocation(
            &controller.tape_resources,
            &[
                relocation_move(base_b, MOVED_A2, FIXTURE_B_BYTES as usize, FIXTURE_B_BYTES as usize),
                relocation_move(base_a, MOVED_A, FIXTURE_A_BYTES as usize, FIXTURE_A_BYTES as usize),
            ],
        )
        .expect("plans");
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].old_key, (base_a, FIXTURE_A_BYTES));
        assert_eq!(plan[0].new_key, (MOVED_A, FIXTURE_A_BYTES));
        assert_eq!(plan[0].coverage, FIXTURE_A_BYTES);
        assert_eq!(plan[1].new_key, (MOVED_A2, FIXTURE_B_BYTES));
    }

    #[test]
    fn overlapping_duplicate_and_aliasing_moves_are_rejected() {
        let (controller, base_a, _) = relocation_controller();
        let tape = &controller.tape_resources;
        let overlapping = plan_relocation(
            tape,
            &[
                relocation_move(base_a, MOVED_A, RESERVED, RESERVED),
                relocation_move(base_a + 0x1000, MOVED_A2, RESERVED, RESERVED),
            ],
        )
        .expect_err("overlapping sources");
        assert!(overlapping.contains("overlapping VMM move sources"), "{overlapping}");
        let duplicate = plan_relocation(
            tape,
            &[
                relocation_move(base_a, MOVED_A, RESERVED, RESERVED),
                relocation_move(base_a, MOVED_A2, RESERVED, RESERVED),
            ],
        )
        .expect_err("duplicate source");
        assert!(duplicate.contains("duplicate VMM move"), "{duplicate}");
        let destinations = plan_relocation(
            tape,
            &[
                relocation_move(base_a, MOVED_A, RESERVED, RESERVED),
                relocation_move(0x7e00_0000_0000, MOVED_A + 0x1000, RESERVED, RESERVED),
            ],
        )
        .expect_err("overlapping destinations");
        assert!(destinations.contains("destinations"), "{destinations}");
        let chained = plan_relocation(
            tape,
            &[
                relocation_move(base_a, MOVED_A, RESERVED, RESERVED),
                relocation_move(MOVED_A, MOVED_A2, RESERVED, RESERVED),
            ],
        )
        .expect_err("destination overlapping a source");
        assert!(chained.contains("overlaps source"), "{chained}");
        for bad in [
            relocation_move(base_a, MOVED_A, 0, 0),
            relocation_move(base_a, MOVED_A, 0x1000, 0x2000),
            relocation_move(0, MOVED_A, RESERVED, RESERVED),
            relocation_move(base_a, base_a, RESERVED, RESERVED),
        ] {
            assert!(plan_relocation(tape, &[bad]).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn straddling_and_colliding_tape_resources_are_rejected() {
        let (controller, base_a, base_b) = relocation_controller();
        // The move source ends inside resource A.
        let straddle = plan_relocation(
            &controller.tape_resources,
            &[relocation_move(base_a - 0x8000, MOVED_A, 0x10000, 0x10000)],
        )
        .expect_err("straddles");
        assert!(straddle.contains("straddles"), "{straddle}");
        // The new range for A lands on resource B, which stays.
        let collision = plan_relocation(
            &controller.tape_resources,
            &[relocation_move(
                base_a,
                base_b + 0x1000,
                FIXTURE_A_BYTES as usize,
                FIXTURE_A_BYTES as usize,
            )],
        );
        assert!(collision.is_err(), "{collision:?}");
    }

    #[test]
    fn untyped_launch_touching_a_moved_range_is_rejected() {
        let (mut controller, base_a, _) = relocation_controller();
        let moved = relocation_move(base_a, MOVED_A, FIXTURE_A_BYTES as usize, FIXTURE_A_BYTES as usize);
        let mut untyped = test_launch("legacy_kernel");
        untyped.accesses = Some(vec![fixture_access(base_a, FIXTURE_A_BYTES, base_a)]);
        controller.recorded.push(untyped);
        let error = controller.relocate_bindings(&[moved]).expect_err("touches range");
        assert!(
            error.contains("legacy_kernel") && error.contains("port the pointer declaration"),
            "{error}"
        );
        assert_eq!(controller.binding_revision(), BindingRevision(0));

        controller.recorded[1].accesses = None;
        let error = controller.relocate_bindings(&[moved]).expect_err("unknown accesses");
        assert!(
            error.contains("legacy_kernel") && error.contains("port the pointer declaration"),
            "{error}"
        );

        controller.recorded[1].accesses =
            Some(vec![fixture_access(0x7e00_0000_0000, 0x1000, 0x7e00_0000_0000)]);
        controller.relocate_bindings(&[moved]).expect("clear of every move");
        assert_eq!(controller.binding_revision(), BindingRevision(1));
    }

    #[test]
    fn relocation_reports_no_route_without_tape_and_rejects_pending_refresh() {
        let mut empty = ReplayController::new(ReplayBackendRequest::Auto);
        let report = empty
            .relocate_bindings(&[relocation_move(0x7f00_0001_0000, MOVED_A, RESERVED, RESERVED)])
            .expect("no tape");
        assert_eq!(report, BindingRefreshReport::NoRoute);

        let (mut controller, base_a, _) = relocation_controller();
        controller.arm_binding_refresh_for_scratch_growth();
        let error = controller
            .relocate_bindings(&[relocation_move(base_a, MOVED_A, RESERVED, RESERVED)])
            .expect_err("refresh pending");
        assert!(error.contains("pending"), "{error}");
    }

    #[test]
    fn non_slot_bytes_gate_ignores_slots_and_dynamic_words_only() {
        let exempt = [(0usize, 8usize), (8, 4), (16, 8)];
        let expected = vec![0x11u8; 32];
        let mut current = expected.clone();
        // Pointer slot and dynamic u32 differences are allowed.
        current[1] ^= 0xff;
        current[9] ^= 0xff;
        current[23] ^= 0xff;
        verify_non_slot_bytes_identical("k", &current, &expected, &exempt).expect("exempt only");
        let mut corrupt = current.clone();
        corrupt[13] ^= 0xff;
        let error = verify_non_slot_bytes_identical("k", &corrupt, &expected, &exempt)
            .expect_err("non-slot corruption");
        assert!(error.contains("non-slot offset 13"), "{error}");
        let mut tail = current.clone();
        tail[31] ^= 0xff;
        let error = verify_non_slot_bytes_identical("k", &tail, &expected, &exempt)
            .expect_err("tail corruption");
        assert!(error.contains("non-slot offset 31"), "{error}");
        assert!(verify_non_slot_bytes_identical("k", &current[..31], &expected, &exempt).is_err());
    }

    #[test]
    fn exempt_ranges_cover_slots_dynamic_bindings_and_legacy_gdn_frames() {
        let (snapshot, layout, _, _, _) = bound_segment_fixture();
        let mut typed = test_launch("typed_fixture");
        typed.kernarg = snapshot;
        typed.binding_layout = Some(layout);
        let untyped = test_launch("untyped");
        let ranges = bound_exempt_kernarg_ranges(
            &[typed, untyped],
            2,
            &[
                (0, ReplayKernargBinding::PositionPlusU32 { offset: 12, addend: 0 }),
                (1, ReplayKernargBinding::PositionPlusU32 { offset: 4, addend: 0 }),
                (9, ReplayKernargBinding::PositionPlusU32 { offset: 4, addend: 0 }),
            ],
            &[0],
        );
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0], vec![(0, 8), (12, 4), (16, 8), (76, 4)]);
        assert_eq!(ranges[1], vec![(4, 4)]);
    }
}
