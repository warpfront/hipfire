// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! railgun's gfx12 (and exact gfx1100) lowering transport, extracted from the M1/M2 shadow
//! (`railgun_shadow::lower`) so a *directly authored* program can use it
//! without any Redline recording (design A §1 "Its emission transport can be
//! reused; its recorder/planner must not be used for this feature", slice
//! A1).
//!
//! * [`emit_segment`] is the one Pm4Op→command emission loop. The legacy M1
//!   shadow (`railgun_shadow::lower`) now calls it with the closure that does
//!   what its inline loop did, so the legacy AR/M2 path emits the same
//!   dwords in the same order.
//! * [`lower_owned`] lowers a single-segment, all-PM4 [`rk::KernelProgram`]
//!   with railgun's own kernarg images and owns everything the queue reads:
//!   loaded kernels, kernarg buffers and the retained indirect buffer. It
//!   needs no `RecordedHipLaunch`, `ReplayBindings` or Redline kernarg
//!   buffer. [`OwnedWindowLowering`] is the GPU-facing half of
//!   `railgun::window::PreparedWindow`: the integration owner implements
//!   `railgun::window::WindowQueue` over it (write images, patch grids,
//!   publish, replay).

use std::collections::BTreeMap;

use railgun::kernel as rk;
use railgun::plan::{self as rp, Pm4Op, Rung};
use redline_dispatch::aql::{load_symbols, Executable, GpuSelector, PciBusId, Runtime};

use super::railgun_shadow::Transport;
use super::{
    gfx10_dispatch_initiator_policy, gfx1151_cu_count_knob_refusal, gfx1151_dispatch_interleave,
    gfx1151_entry_acquire_policy, gfx1151_resource_limits_policy, retained_kernarg_pool,
    Gfx10DispatchInitiatorPolicy, Gfx10Pm4CommandBuffer, Gfx11ComputeResourceLimitsPolicy,
    Gfx11EntryAcquirePolicy, Gfx12DispatchPacing, GpuDevice, KernargBuffer, KernargPool, Kernel,
    LaunchGeometry, LegacyDependencyMode, Pm4Architecture, Pm4Commands, Pm4RegisterPolicy,
    ReplayQuiescence, SingleQueuePm4Ib,
};
use crate::code_object::{CodeObjectId, RecordedArtifact};

/// A fresh command buffer from Redline's builder, paced per `lowering`.
pub(super) fn paced_commands(transport: &Transport<'_>, lowering: rp::ArchLowering) -> Pm4Commands {
    paced(transport.new_commands, lowering)
}

fn paced(new_commands: &dyn Fn() -> Pm4Commands, lowering: rp::ArchLowering) -> Pm4Commands {
    let mut commands = new_commands();
    if let Pm4Commands::Gfx12(gfx12) = &mut commands {
        gfx12.set_dispatch_pacing(match lowering.post_dispatch_nop {
            0 => Gfx12DispatchPacing::None,
            n => Gfx12DispatchPacing::PostDispatchNop(n),
        });
    }
    commands
}

/// Emit one segment's commands: entry acquire, the planned boundaries,
/// `dispatch` for every node, and the trailing release. `dispatch` emits the
/// node's dispatch packet (and owns its kernarg buffer).
pub(super) fn emit_segment(
    plan: &rp::Pm4Plan,
    segment: &rp::Segment,
    arch_lowering: rp::ArchLowering,
    transport: &Transport<'_>,
    dispatch: impl FnMut(&mut Pm4Commands, usize) -> Result<(), String>,
) -> Result<Pm4Commands, String> {
    emit_segment_with(
        plan,
        segment,
        arch_lowering,
        transport.new_commands,
        transport.gcr_trim,
        transport.entry_policy,
        dispatch,
    )
}

/// [`emit_segment`] over exactly the transport parameters it reads, so the
/// emission is exercised without a device. Every rung lowers through the
/// stream's own architecture ([`Pm4Commands::system_acquire`],
/// [`Pm4Commands::acquire_inter_node`]); a pacing request on a stream with no
/// pacing packet (anything but gfx12) is refused, never dropped.
fn emit_segment_with(
    plan: &rp::Pm4Plan,
    segment: &rp::Segment,
    arch_lowering: rp::ArchLowering,
    new_commands: &dyn Fn() -> Pm4Commands,
    gcr_trim: bool,
    entry_policy: Gfx11EntryAcquirePolicy,
    mut dispatch: impl FnMut(&mut Pm4Commands, usize) -> Result<(), String>,
) -> Result<Pm4Commands, String> {
    let mut commands = paced(new_commands, arch_lowering);
    if arch_lowering.post_dispatch_nop != 0 && commands.architecture() != Pm4Architecture::Gfx12 {
        return Err(format!(
            "post-dispatch NOP pacing ({} dwords) is a gfx12 packet; a {:?} stream has none",
            arch_lowering.post_dispatch_nop,
            commands.architecture()
        ));
    }
    for op in plan.segment_ops(segment) {
        match op {
            Pm4Op::Entry => {
                commands.emit_entry_sentinel_reset()?;
                commands.acquire_entry(gcr_trim, entry_policy);
            }
            Pm4Op::Boundary(v) => {
                if v.wait_idle {
                    commands.wait_compute_idle()?;
                }
                match v.acquire {
                    Rung::None => {}
                    Rung::InterNode => commands.acquire_inter_node(gcr_trim, false),
                    Rung::System => commands.system_acquire(),
                }
            }
            Pm4Op::Dispatch(i) => dispatch(&mut commands, i)?,
            Pm4Op::Exit => commands.trailing_release()?,
            // Recorded tapes, DFlash windows and owned windows author no
            // conditionals: gfx12 has no COND_EXEC receipt and gfx1100 has
            // none either (`rp::cp_skip_admitted`).
            Pm4Op::SkipBegin(_) | Pm4Op::SkipEnd => {
                return Err("railgun M1 lowers no CP-skip conditionals".into())
            }
        }
    }
    Ok(commands)
}

/// What [`check_gfx11_stream`] counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Gfx11StreamCensus {
    pub(crate) dispatches: usize,
    pub(crate) wait_idles: usize,
    /// Mid-segment `acquire_inter_node_same_agent` (GCR 0x380) packets.
    pub(crate) inter_node_acquires: usize,
    /// ROCr system `ACQUIRE_MEM` packets (entry, boundaries, exit).
    pub(crate) system_acquires: usize,
    pub(crate) register_runs: usize,
}

/// Gate on the lowered exact-gfx1100 command stream: it may hold only what
/// the gfx11 legacy builder emits for railgun's rungs and nothing from the
/// gfx12 or conditional vocabularies. Accepted: `SET_SH_REG` (never the
/// `COMPUTE_DISPATCH_INTERLEAVE` register: no interleave experiment is part
/// of the lowering), `DISPATCH_DIRECT`, `EVENT_WRITE CS_PARTIAL_FLUSH`, and
/// `ACQUIRE_MEM` that is byte-identical to the builder's own system acquire
/// or same-agent acquire (GCR 0x380). A `NOP` (gfx12 pacing), `COND_EXEC`,
/// `RELEASE_MEM`/`WAIT_REG_MEM` fences, any other opcode, or an `ACQUIRE_MEM`
/// with a gfx12 GCR value fails the lowering naming the dword.
pub(crate) fn check_gfx11_stream(dwords: &[u32]) -> Result<Gfx11StreamCensus, String> {
    const SET_SH_REG: u32 = 0x76;
    const DISPATCH_DIRECT: u32 = 0x15;
    const EVENT_WRITE: u32 = 0x46;
    const ACQUIRE_MEM: u32 = 0x58;
    const CS_PARTIAL_FLUSH: u32 = 0x407;
    const COMPUTE_DISPATCH_INTERLEAVE: u32 = 0x22f;
    let reference = |emit: fn(&mut Gfx10Pm4CommandBuffer)| {
        let mut commands = Gfx10Pm4CommandBuffer::new();
        emit(&mut commands);
        commands.dwords().to_vec()
    };
    let system = reference(|c| c.acquire_system());
    let same_agent = reference(|c| c.acquire_inter_node_same_agent());
    let mut census = Gfx11StreamCensus::default();
    let mut cursor = 0usize;
    while cursor < dwords.len() {
        let header = dwords[cursor];
        if header >> 30 != 3 {
            return Err(format!(
                "dword {cursor} {header:#010x} is not a type-3 packet"
            ));
        }
        let opcode = (header >> 8) & 0xff;
        let body_dwords = ((header >> 16) & 0x3fff) as usize + 1;
        let next = cursor + 1 + body_dwords;
        if next > dwords.len() {
            return Err(format!(
                "dword {cursor}: packet {opcode:#x} overruns the stream"
            ));
        }
        let body = &dwords[cursor + 1..next];
        match opcode {
            SET_SH_REG => {
                let first = body[0];
                let end = first + (body.len() as u32 - 1);
                if (first..end).contains(&COMPUTE_DISPATCH_INTERLEAVE) {
                    return Err(format!(
                        "dword {cursor}: SET_SH_REG writes COMPUTE_DISPATCH_INTERLEAVE"
                    ));
                }
                census.register_runs += 1;
            }
            DISPATCH_DIRECT => {
                if body_dwords != 4 {
                    return Err(format!(
                        "dword {cursor}: DISPATCH_DIRECT has {body_dwords} body dwords"
                    ));
                }
                census.dispatches += 1;
            }
            EVENT_WRITE => {
                if body != [CS_PARTIAL_FLUSH] {
                    return Err(format!(
                        "dword {cursor}: EVENT_WRITE {body:#x?} is not CS_PARTIAL_FLUSH"
                    ));
                }
                census.wait_idles += 1;
            }
            ACQUIRE_MEM => {
                let packet = &dwords[cursor..next];
                if packet == system.as_slice() {
                    census.system_acquires += 1;
                } else if packet == same_agent.as_slice() {
                    census.inter_node_acquires += 1;
                } else {
                    return Err(format!(
                        "dword {cursor}: ACQUIRE_MEM {packet:#x?} is neither the gfx11 system \
                         acquire nor the same-agent acquire (GCR 0x380)"
                    ));
                }
            }
            0x10 => {
                return Err(format!(
                    "dword {cursor}: NOP packet (gfx12 dispatch pacing) in a gfx11 stream"
                ))
            }
            0x22 => return Err(format!("dword {cursor}: COND_EXEC in a gfx11 owned stream")),
            other => {
                return Err(format!(
                    "dword {cursor}: PM4 opcode {other:#x} is not part of the gfx11 owned lowering"
                ))
            }
        }
        cursor = next;
    }
    Ok(census)
}

/// A dispatch packet's native dimensions for a HIP grid of `grid` workgroups
/// of `block` threads: gfx12 `USE_THREAD_DIMENSIONS` counts work-items
/// (`Err(axis)` on `u32` overflow); the gfx10/gfx11 packet counts
/// workgroups, so the HIP grid is the packet value unscaled.
fn native_dimensions(
    architecture: Pm4Architecture,
    grid: [u32; 3],
    block: [u32; 3],
) -> Result<[u32; 3], usize> {
    if architecture != Pm4Architecture::Gfx12 {
        return Ok(grid);
    }
    let mut workitems = [0u32; 3];
    for axis in 0..3 {
        workitems[axis] = grid[axis].checked_mul(block[axis]).ok_or(axis)?;
    }
    Ok(workitems)
}

/// One kernarg buffer whose bytes differ from the bound image.
#[derive(Clone, Debug)]
pub(crate) struct KernargMismatch {
    pub(crate) node: usize,
    pub(crate) symbol: String,
    /// VA of the buffer, its length and the pool allocation extent it owns.
    pub(crate) address: u64,
    pub(crate) length: usize,
    pub(crate) allocation: usize,
    /// First differing byte and the dwords around it.
    pub(crate) offset: usize,
    pub(crate) expected: u32,
    pub(crate) actual: u32,
    pub(crate) differing_bytes: usize,
    pub(crate) expected_head: Vec<u8>,
    pub(crate) actual_head: Vec<u8>,
}

/// A failed pre-submit self-check of the lowered state.
#[derive(Clone, Debug)]
pub(crate) struct SelfCheckFailure {
    /// The one-line description of the first mismatch.
    pub(crate) first: String,
    /// Every kernarg buffer that differs from its bound image.
    pub(crate) kernarg: Vec<KernargMismatch>,
}

/// A failed replay and what is known of the queue afterwards.
#[derive(Clone, Debug)]
pub(crate) struct OwnedReplayFailure {
    pub error: String,
    pub quiescence: ReplayQuiescence,
}

/// railgun's lowering of a directly authored program, with every object the
/// queue reads owned. Field order is drop order: the indirect buffer first,
/// then the kernels it dispatches, then the kernarg buffers it points at
/// (the order `PreparedPm4Replay` uses).
pub(crate) struct OwnedWindowLowering {
    graph: SingleQueuePm4Ib,
    _kernels: Vec<Kernel>,
    kernargs: Vec<KernargBuffer>,
    blocks: Vec<[u32; 3]>,
    publish: Option<usize>,
    command_dwords: u32,
    /// The stream's PM4 architecture: decides the dispatch packet's native
    /// dimension unit (gfx12 work-items, gfx11 workgroups).
    architecture: Pm4Architecture,
}

/// Lower `program` (one segment spanning every node, every node PM4) through
/// railgun's own plan with railgun's kernarg images. `kernels[i]` is the
/// loaded object of node `i`. Kernarg buffers come from `transport.pool`
/// (Redline's retained pool); the indirect buffer and its host-read
/// completion surfaces come from `graph_pool` (Redline's host pool). Nothing
/// is submitted.
fn lower_owned(
    program: &rk::KernelProgram,
    plan: &rp::Pm4Plan,
    arch_lowering: rp::ArchLowering,
    architecture: Pm4Architecture,
    kernels: Vec<Kernel>,
    transport: &Transport<'_>,
    graph_pool: &KernargPool,
    device: &GpuDevice,
) -> Result<OwnedWindowLowering, String> {
    let nodes = program.nodes.len();
    if !plan.is_single_segment(nodes) {
        return Err(format!(
            "an owned window lowers exactly one segment over {nodes} nodes ({} segments)",
            plan.segments.len()
        ));
    }
    if let Some((i, node)) = program
        .nodes
        .iter()
        .enumerate()
        .find(|(_, n)| !n.lowering.is_pm4())
    {
        return Err(format!(
            "node {i} {} is not PM4-lowered: {:?}",
            node.symbol, node.lowering
        ));
    }
    if kernels.len() != nodes {
        return Err(format!(
            "{} loaded kernels for {nodes} nodes",
            kernels.len()
        ));
    }
    let words = program.authored_words();
    let mut buffers: Vec<Option<KernargBuffer>> = (0..nodes).map(|_| None).collect();
    let commands = emit_segment(
        plan,
        &plan.segments[0],
        arch_lowering,
        transport,
        |commands, i| {
            let node = &program.nodes[i];
            let metadata = kernels[i].metadata();
            let image = program.kernarg_image(i, metadata.kernarg_segment_size as usize, &words)?;
            let mut buffer = transport
                .pool
                .allocate_for(metadata)
                .map_err(|e| format!("{}: kernarg: {e}", node.symbol))?;
            buffer
                .write_exact(&image)
                .map_err(|e| format!("{}: kernarg: {e}", node.symbol))?;
            let mut workgroup = [0u16; 3];
            for axis in 0..3 {
                workgroup[axis] = u16::try_from(node.block[axis])
                    .map_err(|_| format!("{}: block exceeds u16", node.symbol))?;
            }
            let geometry = LaunchGeometry::from_hip_workgroups(node.grid, workgroup)
                .map_err(|e| format!("{}: {e}", node.symbol))?;
            device
                .validate_geometry(geometry)
                .map_err(|e| format!("{}: {e}", node.symbol))?;
            commands
                .dispatch(&kernels[i], geometry, node.dynamic_lds, buffer.address())
                .map_err(|e| format!("{}: {e}", node.symbol))?;
            buffers[i] = Some(buffer);
            Ok(())
        },
    )?;
    // The stream must be the one the context's device architecture lowers:
    // gfx12 command dwords, or a gfx11 stream holding only the vocabulary
    // exact gfx1100 owns (no gfx12 packet, no pacing NOP, no COND_EXEC).
    if commands.architecture() != architecture {
        return Err(format!(
            "owned lowering built a {:?} stream for a {architecture:?} device",
            commands.architecture()
        ));
    }
    match architecture {
        Pm4Architecture::Gfx12 => {
            if commands.gfx12_dwords().is_none() {
                return Err("an owned window lowers gfx12 command buffers only".to_owned());
            }
        }
        Pm4Architecture::Gfx11 => {
            let dwords = commands
                .gfx11_dwords()
                .ok_or("an owned gfx11 window lowers gfx11 command buffers only")?;
            check_gfx11_stream(dwords)
                .map_err(|reason| format!("owned gfx1100 command stream refused: {reason}"))?;
        }
        Pm4Architecture::Gfx10 => return Err("an owned window never lowers gfx10".to_owned()),
    }
    let kernargs: Vec<KernargBuffer> = buffers
        .into_iter()
        .enumerate()
        .map(|(i, b)| b.ok_or_else(|| format!("node {i} was never dispatched")))
        .collect::<Result<_, _>>()?;
    let command_dwords = commands.len_dwords();
    let graph = commands.create_graph(device, graph_pool, None, false)?;
    let publish = kernargs
        .iter()
        .rposition(|kernarg| kernarg.is_device_local() && !kernarg.is_empty());
    Ok(OwnedWindowLowering {
        graph,
        _kernels: kernels,
        kernargs,
        blocks: program.nodes.iter().map(|n| n.block).collect(),
        publish,
        command_dwords,
        architecture,
    })
}

/// How [`LoweringContext::new`] picks the ROCr device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeviceSelect {
    /// The HSA agent at this ordinal (gfx12 owners; HIP and ROCr ordinals
    /// agree on the gfx12 hosts it is certified on).
    Ordinal(usize),
    /// The one ROCr GPU agent at this PCI bus address (the HIP device's own
    /// BDF). Required for gfx1100: the 7900 XTX hosts are multi-GPU and
    /// ordinals are not a device identity.
    Pci(PciBusId),
}

/// Everything an owned lowering needs that does not depend on one window,
/// built ONCE per owner (the window builder) and shared by every skeleton it
/// lowers: one HSA [`Runtime`] and [`GpuDevice`], the host pool (indirect
/// buffers, completion surfaces), the retained kernarg pool (Redline's, VRAM by
/// default) and every loaded code object. A skeleton's kernels, kernarg buffers
/// and indirect buffer each hold an `Arc` to what they use (kernarg buffer ->
/// pool -> runtime, kernel -> executable -> runtime), so they outlive this
/// context and are released only when the skeleton is dropped.
pub(crate) struct LoweringContext {
    // Field order is drop order: loaded objects before the pools before the
    // device and runtime they belong to.
    executables: BTreeMap<CodeObjectId, Executable>,
    resolved: BTreeMap<(CodeObjectId, String), Kernel>,
    kernarg_pool: KernargPool,
    host_pool: KernargPool,
    device: GpuDevice,
    architecture: Pm4Architecture,
    register_policy: Pm4RegisterPolicy,
    dispatch_initiator_policy: super::Gfx10DispatchInitiatorPolicy,
    dispatch_interleave: Option<super::Gfx11DispatchInterleave>,
    resource_limits_policy: super::Gfx11ComputeResourceLimitsPolicy,
    entry_policy: super::Gfx11EntryAcquirePolicy,
    gcr_trim: bool,
    _runtime: Runtime,
}

/// Why an owned lowering refuses the ROCr device named `name` on PM4
/// architecture `architecture` and selector `select`, or `None` when it is a
/// certified target: any gfx12 device by ordinal or PCI address, and EXACT
/// `gfx1100` identified by PCI address. Every other gfx11 part (gfx1151
/// first) and gfx10 are refused.
fn device_refusal(
    architecture: Pm4Architecture,
    name: &str,
    select: DeviceSelect,
) -> Option<String> {
    match architecture {
        Pm4Architecture::Gfx12 => None,
        Pm4Architecture::Gfx11 if name == "gfx1100" => match select {
            DeviceSelect::Pci(_) => None,
            DeviceSelect::Ordinal(ordinal) => Some(format!(
                "an owned gfx1100 window needs the device's PCI identity; ordinal {ordinal} \
                 does not identify a card on a multi-GPU host"
            )),
        },
        Pm4Architecture::Gfx11 => Some(format!(
            "an owned window lowers gfx12 or exact gfx1100 command buffers; gfx11 device {name} \
             is refused (railgun has no cache table, probe receipt or lowering certification for it)"
        )),
        Pm4Architecture::Gfx10 => Some(format!(
            "an owned window lowers gfx12 or exact gfx1100 command buffers; device {name} is gfx10"
        )),
    }
}

impl LoweringContext {
    /// The context of the device `select` picks with the transport parameters
    /// Redline's PM4 prepare uses (`ReplayController::prepare_pm4_prefix_inner`):
    /// the dispatch-initiator, interleave, resource-limits and entry-acquire
    /// policies, the retained kernarg pool, the `HIPFIRE_REPLAY_PM4_GCR_TRIM`
    /// setting and `LegacyDependencyMode::CsPartialFlush`.
    ///
    /// gfx12 is selected by the caller's ordinal or PCI address. Exact
    /// gfx1100 requires the PCI address (a single matching ROCr GPU agent
    /// named exactly `gfx1100`) and takes the legacy defaults of every gfx11
    /// policy WITHOUT reading the gfx1151 experiment knobs
    /// (`HIPFIRE_GFX1151_PM4_*`, `HIPFIRE_GFX1100_PM4_EXPERIMENTS`): the
    /// lowering railgun certifies on gfx1100 is the byte-for-byte legacy one.
    pub(crate) fn new(select: DeviceSelect) -> Result<Self, String> {
        let runtime = Runtime::initialize(load_symbols().map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        let device = match select {
            DeviceSelect::Ordinal(ordinal) => runtime
                .select_gpu(GpuSelector::Ordinal(ordinal))
                .map_err(|error| error.to_string())?,
            DeviceSelect::Pci(pci) => {
                let mut matches = runtime
                    .gpu_devices()
                    .map_err(|error| error.to_string())?
                    .into_iter()
                    .filter(|device| device.pci_bus_id() == pci);
                let device = matches
                    .next()
                    .ok_or_else(|| format!("no ROCr GPU agent at PCI {pci}"))?;
                if matches.next().is_some() {
                    return Err(format!("more than one ROCr GPU agent at PCI {pci}"));
                }
                device
            }
        };
        let architecture = Pm4Architecture::from_device(&device)?;
        if let Some(reason) = device_refusal(architecture, device.name(), select) {
            return Err(reason);
        }
        gfx1151_cu_count_knob_refusal()?;
        let (dispatch_initiator_policy, dispatch_interleave, resource_limits_policy, entry_policy) =
            if architecture == Pm4Architecture::Gfx11 {
                (
                    Gfx10DispatchInitiatorPolicy::Legacy,
                    None,
                    Gfx11ComputeResourceLimitsPolicy::Legacy,
                    Gfx11EntryAcquirePolicy::System,
                )
            } else {
                (
                    gfx10_dispatch_initiator_policy(architecture, device.name()),
                    gfx1151_dispatch_interleave(architecture, device.name()),
                    gfx1151_resource_limits_policy(architecture, device.name()),
                    gfx1151_entry_acquire_policy(architecture, device.name()),
                )
            };
        let host_pool = KernargPool::discover(&device).map_err(|error| error.to_string())?;
        let kernarg_pool = retained_kernarg_pool(&device, &host_pool, entry_policy);
        let register_policy = Pm4RegisterPolicy::from_config();
        let gcr_trim = hipfire_config::process_value("HIPFIRE_REPLAY_PM4_GCR_TRIM")
            .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
            .unwrap_or(true);
        Ok(Self {
            executables: BTreeMap::new(),
            resolved: BTreeMap::new(),
            kernarg_pool,
            host_pool,
            device,
            architecture,
            register_policy,
            dispatch_initiator_policy,
            dispatch_interleave,
            resource_limits_policy,
            entry_policy,
            gcr_trim,
            _runtime: runtime,
        })
    }

    /// Distinct code objects loaded so far (each is loaded once per context).
    pub(crate) fn loaded_objects(&self) -> usize {
        self.executables.len()
    }

    /// [`lower_owned`] over this context. `artifacts[i]` is the loaded code
    /// object that owns node `i`'s symbol; each distinct object is loaded once
    /// for the whole context and node `i` resolves `{symbol}.kd` in it.
    /// Nothing is submitted.
    pub(crate) fn lower(
        &mut self,
        program: &rk::KernelProgram,
        plan: &rp::Pm4Plan,
        arch_lowering: rp::ArchLowering,
        artifacts: &[RecordedArtifact],
    ) -> Result<(OwnedWindowLowering, LowerTiming), String> {
        let started = std::time::Instant::now();
        let loaded_before = self.executables.len();
        if artifacts.len() != program.nodes.len() {
            return Err(format!(
                "{} code objects for {} nodes",
                artifacts.len(),
                program.nodes.len()
            ));
        }
        // Exact-device identity (gfx1100): the program must have been authored
        // for the very architecture of the device this context lowers for.
        if self.architecture == Pm4Architecture::Gfx11 && program.arch != self.device.name() {
            return Err(format!(
                "program arch {} is not the owned lowering device {}",
                program.arch,
                self.device.name()
            ));
        }
        let mut kernels = Vec::with_capacity(program.nodes.len());
        for (node, artifact) in program.nodes.iter().zip(artifacts) {
            let id = artifact.id();
            if !self.executables.contains_key(&id) {
                let executable = Executable::load(&self.device, artifact.image().clone())
                    .map_err(|error| format!("load {artifact}: {error}"))?;
                self.executables.insert(id, executable);
            }
            let key = (id, format!("{}.kd", node.symbol));
            if !self.resolved.contains_key(&key) {
                let kernel = self.executables[&id]
                    .kernel(&key.1)
                    .map_err(|error| format!("resolve {}: {error}", key.1))?;
                self.resolved.insert(key.clone(), kernel);
            }
            kernels.push(self.resolved[&key].clone());
        }
        let load_us = started.elapsed().as_micros() as u64;
        let new_objects = self.executables.len() - loaded_before;
        let (architecture, register_policy) = (self.architecture, self.register_policy);
        let (initiator, interleave, limits) = (
            self.dispatch_initiator_policy,
            self.dispatch_interleave,
            self.resource_limits_policy,
        );
        let new_commands = || {
            Pm4Commands::new_with_dependency(
                architecture,
                register_policy,
                initiator,
                interleave,
                limits,
                LegacyDependencyMode::CsPartialFlush,
            )
        };
        let transport = Transport {
            pool: &self.kernarg_pool,
            new_commands: &new_commands,
            gcr_trim: self.gcr_trim,
            entry_policy: self.entry_policy,
        };
        let emit_started = std::time::Instant::now();
        let lowering = lower_owned(
            program,
            plan,
            arch_lowering,
            architecture,
            kernels,
            &transport,
            &self.host_pool,
            &self.device,
        )?;
        Ok((
            lowering,
            LowerTiming {
                load_us,
                new_objects,
                emit_us: emit_started.elapsed().as_micros() as u64,
            },
        ))
    }
}

/// Where one [`LoweringContext::lower`] spent its time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowerTiming {
    /// Loading and resolving code objects (zero once they are all loaded).
    pub load_us: u64,
    /// Code objects loaded by this call (not already in the context).
    pub new_objects: usize,
    /// Kernarg allocation, IB emission and graph creation.
    pub emit_us: u64,
}

impl OwnedWindowLowering {
    /// Command dwords of the retained indirect buffer.
    pub(crate) fn command_dwords(&self) -> u32 {
        self.command_dwords
    }

    /// Replace node `node`'s whole kernarg segment (one
    /// `KernelProgram::encode_kernarg_into` image). Only valid while the
    /// queue is quiescent.
    pub(crate) fn write_kernarg(&mut self, node: usize, image: &[u8]) -> Result<(), String> {
        self.kernargs
            .get_mut(node)
            .ok_or_else(|| format!("no node {node}"))?
            .write_exact(image)
            .map_err(|e| format!("node {node}: kernarg: {e}"))
    }

    /// Patch node `node`'s dispatch geometry in the indirect buffer to the
    /// HIP grid `grid` (workgroups). gfx12 dispatch packets count work-items,
    /// so the grid is scaled by the node's block, checked; the gfx11
    /// `DISPATCH_DIRECT` counts workgroups and takes the grid as is. Only
    /// valid while the queue is quiescent.
    pub(crate) fn patch_grid(&mut self, node: usize, grid: [u32; 3]) -> Result<(), String> {
        let block = *self
            .blocks
            .get(node)
            .ok_or_else(|| format!("no node {node}"))?;
        let dimensions = native_dimensions(self.architecture, grid, block).map_err(|axis| {
            format!(
                "node {node}: grid overflow axis={axis} workgroups={} workgroup={}",
                grid[axis], block[axis]
            )
        })?;
        self.graph
            .patch_dispatch_dimensions(node, dimensions)
            .map_err(|e| e.to_string())
    }

    /// Read back the lowered state exactly as the command processor and the
    /// kernels will see it and compare it with what the window bound: every
    /// node's kernarg buffer (explicit and hidden bytes) against its bound
    /// image, and every `DISPATCH_DIRECT` in the retained indirect buffer
    /// against the bound grid (work-items) and the node's block. Quiescent
    /// use only. Every kernarg buffer is scanned (after a publish, so a
    /// posted-but-unlanded write cannot hide); the failure names the first
    /// mismatch and carries every mismatching buffer with its VA range.
    pub(crate) fn verify_state(
        &mut self,
        symbols: &[String],
        images: &[&[u8]],
        grids: &[[u32; 3]],
    ) -> Result<(), SelfCheckFailure> {
        let nodes = self.kernargs.len();
        let plain = |first: String| SelfCheckFailure {
            first,
            kernarg: Vec::new(),
        };
        if images.len() != nodes || grids.len() != nodes || symbols.len() != nodes {
            return Err(plain(format!(
                "{} images, {} grids, {} symbols for {nodes} lowered nodes",
                images.len(),
                grids.len(),
                symbols.len()
            )));
        }
        self.publish_host_writes();
        let mut bad: Vec<KernargMismatch> = Vec::new();
        let mut shape_error: Option<String> = None;
        for (node, (buffer, image)) in self.kernargs.iter_mut().zip(images).enumerate() {
            let address = buffer.address() as usize as u64;
            let allocation = buffer.allocation_size();
            let actual: &[u8] = buffer.as_mut_bytes();
            if actual.len() != image.len() {
                shape_error.get_or_insert_with(|| {
                    format!(
                        "node {node} {}: kernarg buffer is {} bytes, the bound image {}",
                        symbols[node],
                        actual.len(),
                        image.len()
                    )
                });
                continue;
            }
            let Some(at) = actual.iter().zip(*image).position(|(a, b)| a != b) else {
                continue;
            };
            let word = at & !3;
            let get = |bytes: &[u8]| {
                let mut w = [0u8; 4];
                let end = (word + 4).min(bytes.len());
                w[..end - word].copy_from_slice(&bytes[word..end]);
                u32::from_le_bytes(w)
            };
            let head = image.len().min(16);
            bad.push(KernargMismatch {
                node,
                symbol: symbols[node].clone(),
                address,
                length: actual.len(),
                allocation,
                offset: at,
                expected: get(image),
                actual: get(actual),
                differing_bytes: actual.iter().zip(*image).filter(|(a, b)| a != b).count(),
                expected_head: image[..head].to_vec(),
                actual_head: actual[..head].to_vec(),
            });
        }
        if let Some(first) = shape_error {
            return Err(SelfCheckFailure {
                first,
                kernarg: bad,
            });
        }
        if let Some(m) = bad.first() {
            return Err(SelfCheckFailure {
                first: format!(
                    "node {} {}: kernarg byte +{} (dword +{}) expected {:#010x} actual {:#010x} \
                     ({} of {} kernarg buffers differ)",
                    m.node,
                    m.symbol,
                    m.offset,
                    m.offset & !3,
                    m.expected,
                    m.actual,
                    bad.len(),
                    nodes
                ),
                kernarg: bad,
            });
        }
        let decoded = self
            .graph
            .decode_dispatches()
            .map_err(|e| plain(e.to_string()))?;
        if decoded.len() != nodes {
            return Err(plain(format!(
                "the retained IB holds {} dispatches for {nodes} lowered nodes",
                decoded.len()
            )));
        }
        for (node, d) in decoded.iter().enumerate() {
            let block = self.blocks[node];
            let want = native_dimensions(self.architecture, grids[node], block).map_err(|_| {
                plain(format!(
                    "node {node} {}: bound grid {:?} x block {block:?} overflows",
                    symbols[node], grids[node]
                ))
            })?;
            if d.dimensions != want {
                let unit = if self.architecture == Pm4Architecture::Gfx12 {
                    format!("grid {:?} x block {block:?}", grids[node])
                } else {
                    format!("workgroup counts {:?}", grids[node])
                };
                return Err(plain(format!(
                    "node {node} {}: IB dispatch dimensions expected {want:?} ({unit}) actual {:?}",
                    symbols[node], d.dimensions
                )));
            }
            if d.workgroup != Some(block) {
                return Err(plain(format!(
                    "node {node} {}: IB workgroup size expected {block:?} actual {:?}",
                    symbols[node], d.workgroup
                )));
            }
        }
        Ok(())
    }

    /// `(address, length, allocation size)` of every node's kernarg buffer.
    pub(crate) fn kernarg_ranges(&self) -> Vec<(u64, usize, usize)> {
        self.kernargs
            .iter()
            .map(|k| (k.address() as usize as u64, k.len(), k.allocation_size()))
            .collect()
    }

    /// Make every host kernarg write since the last call GPU-visible.
    pub(crate) fn publish_host_writes(&self) {
        if let Some(index) = self.publish {
            self.kernargs[index].publish_host_writes();
        }
    }

    /// Submit the retained indirect buffer once and wait for it.
    ///
    /// # Safety
    ///
    /// Every pointer in the written kernarg images must still refer to a live
    /// allocation of the owner (the window binding key proves it).
    pub(crate) unsafe fn replay_and_wait(&mut self) -> Result<(), OwnedReplayFailure> {
        // SAFETY: forwarded from this method's caller.
        unsafe { self.graph.replay_and_wait_checked() }.map_err(Self::replay_failure)
    }

    /// Publish the retained indirect buffer once and return without waiting.
    /// `Ok(())` means the replay is in flight: no kernarg or grid may be
    /// written and nothing may be submitted until [`Self::wait_replay`]
    /// returns (or this lowering is dropped, which inactivates the queue
    /// before the kernargs are freed).
    ///
    /// # Safety
    ///
    /// As [`Self::replay_and_wait`], for the whole time the replay is in
    /// flight (until `wait_replay` returns `Ok`, and after an error through
    /// this lowering's destruction).
    pub(crate) unsafe fn replay_submit(&mut self) -> Result<(), OwnedReplayFailure> {
        // SAFETY: forwarded from this method's caller.
        unsafe { self.graph.replay_submit_checked() }.map_err(Self::replay_failure)
    }

    /// Wait for the replay [`Self::replay_submit`] published (`Ok(())` at once
    /// when none is in flight), with `replay_and_wait`'s timeout, watchdog and
    /// quiescence semantics.
    pub(crate) fn wait_replay(&mut self) -> Result<(), OwnedReplayFailure> {
        self.graph.wait_replay_checked().map_err(Self::replay_failure)
    }

    fn replay_failure(
        (error, quiescence): (
            redline_dispatch::aql::ReplayError,
            redline_dispatch::aql::Quiescence,
        ),
    ) -> OwnedReplayFailure {
        OwnedReplayFailure {
            error: error.to_string(),
            quiescence: match quiescence {
                redline_dispatch::aql::Quiescence::Proven => ReplayQuiescence::Proven,
                redline_dispatch::aql::Quiescence::Unknown => ReplayQuiescence::Unknown,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use railgun::plan::{Boundary, Segment, Visibility, ENTRY};
    use redline_dispatch::aql::{Gfx10KernelImage, Gfx11DispatchInterleave, Gfx12Pm4CommandBuffer};

    const DEPENDENT: Visibility = Visibility {
        wait_idle: true,
        acquire: Rung::InterNode,
        pre_writer: false,
    };
    const STRICT: Visibility = Visibility {
        wait_idle: true,
        acquire: Rung::System,
        pre_writer: false,
    };

    fn new_commands(architecture: Pm4Architecture) -> Pm4Commands {
        Pm4Commands::new_with_dependency(
            architecture,
            Pm4RegisterPolicy::Static,
            Gfx10DispatchInitiatorPolicy::Legacy,
            None,
            Gfx11ComputeResourceLimitsPolicy::Legacy,
            LegacyDependencyMode::CsPartialFlush,
        )
    }

    /// A hand-built single-segment plan: `ENTRY`, then `boundaries`, one node
    /// per boundary (node 0's boundary is the entry).
    fn plan_of(boundaries: &[Visibility]) -> rp::Pm4Plan {
        let mut all = vec![ENTRY];
        all.extend_from_slice(boundaries);
        rp::Pm4Plan {
            policy: rp::RowPolicy::UnreceiptedSystem,
            segments: vec![Segment {
                first: 0,
                end: all.len(),
            }],
            boundaries: all
                .into_iter()
                .map(|visibility| Boundary {
                    visibility,
                    edges: Vec::new(),
                    split: false,
                })
                .collect(),
            rows_used: BTreeMap::new(),
            skips: Vec::new(),
            groups: Vec::new(),
        }
    }

    fn image() -> Gfx10KernelImage {
        Gfx10KernelImage {
            code_entry: 0x1_0000_0000,
            compute_pgm_rsrc1: 0x60af_0000,
            compute_pgm_rsrc2: 0x0000_0084,
            compute_pgm_rsrc3: 0,
            group_segment_size: 0,
            private_segment_size: 0,
            dynamic_callstack: false,
            wave32: true,
            kernel_code_properties: 1 << 10,
        }
    }

    const GRIDS: [[u32; 3]; 3] = [[5, 2, 1], [96, 1, 1], [7, 1, 1]];

    fn lower_gfx11(
        plan: &rp::Pm4Plan,
        arch_lowering: rp::ArchLowering,
    ) -> Result<Pm4Commands, String> {
        let make = || new_commands(Pm4Architecture::Gfx11);
        emit_segment_with(
            plan,
            &plan.segments[0],
            arch_lowering,
            &make,
            true,
            Gfx11EntryAcquirePolicy::System,
            |commands, i| {
                let Pm4Commands::Legacy { commands, .. } = commands else {
                    return Err("not a legacy stream".into());
                };
                let geometry = LaunchGeometry::from_hip_workgroups(GRIDS[i], [256, 1, 1]).unwrap();
                commands
                    .dispatch_image(&image(), geometry, 0, &[0x1000 + i as u32, 0])
                    .map_err(|e| e.to_string())
            },
        )
    }

    /// `(opcode, body)` of every packet.
    fn packets(dwords: &[u32]) -> Vec<(u32, Vec<u32>)> {
        let mut out = Vec::new();
        let mut at = 0;
        while at < dwords.len() {
            let header = dwords[at];
            assert_eq!(header >> 30, 3, "type-3 packet at dword {at}");
            let body = ((header >> 16) & 0x3fff) as usize + 1;
            out.push(((header >> 8) & 0xff, dwords[at + 1..at + 1 + body].to_vec()));
            at += 1 + body;
        }
        out
    }

    #[test]
    fn gfx1100_lowering_is_gfx11_legacy_pm4_only() {
        let plan = plan_of(&[DEPENDENT, STRICT]);
        let commands = lower_gfx11(&plan, rp::ArchLowering::for_arch("gfx1100")).unwrap();
        assert_eq!(commands.architecture(), Pm4Architecture::Gfx11);
        assert!(commands.gfx12_dwords().is_none());
        let dwords = commands.gfx11_dwords().unwrap();
        let census = check_gfx11_stream(dwords).unwrap();
        assert_eq!(
            census,
            Gfx11StreamCensus {
                dispatches: 3,
                // boundary 1, boundary 2 and the exit.
                wait_idles: 3,
                inter_node_acquires: 1,
                // entry, boundary 2 and the exit.
                system_acquires: 3,
                register_runs: census.register_runs,
            }
        );
        assert!(census.register_runs > 0);
        let packets = packets(dwords);
        // No NOP pacing, no COND_EXEC, no fence, nothing but the gfx11 set.
        for (opcode, _) in &packets {
            assert!(
                matches!(opcode, 0x76 | 0x15 | 0x46 | 0x58),
                "unexpected opcode {opcode:#x}"
            );
        }
        // Dispatch packets carry WORKGROUP counts: the HIP grid, unscaled.
        let dispatches: Vec<[u32; 3]> = packets
            .iter()
            .filter(|(opcode, _)| *opcode == 0x15)
            .map(|(_, body)| [body[0], body[1], body[2]])
            .collect();
        assert_eq!(dispatches, GRIDS);
        // The packet order is entry, d0, wait, same-agent, d1, wait, system, d2, exit.
        let shape: Vec<u32> = packets
            .iter()
            .map(|(opcode, _)| *opcode)
            .filter(|opcode| matches!(opcode, 0x15 | 0x46 | 0x58))
            .collect();
        assert_eq!(
            shape,
            [0x58, 0x15, 0x46, 0x58, 0x15, 0x46, 0x58, 0x15, 0x46, 0x58]
        );
        // GCR of every non-system ACQUIRE_MEM is 0x380, never a gfx12 value.
        let gcr: Vec<u32> = packets
            .iter()
            .filter(|(opcode, body)| *opcode == 0x58 && body[5] == 10)
            .map(|(_, body)| body[6])
            .collect();
        assert_eq!(gcr, [0x380]);
    }

    #[test]
    fn gfx1100_lowering_refuses_pacing_it_cannot_emit() {
        let plan = plan_of(&[DEPENDENT]);
        let err = lower_gfx11(
            &plan,
            rp::ArchLowering {
                post_dispatch_nop: 64,
            },
        )
        .err()
        .expect("pacing is a gfx12 packet");
        assert!(err.contains("gfx12 packet"), "{err}");
    }

    #[test]
    fn gfx1100_lowering_refuses_cp_skip() {
        let mut plan = plan_of(&[DEPENDENT]);
        plan.skips.push(railgun::kernel::Conditional {
            flag: railgun::ResourceId(0),
            body: 1..2,
            lowering: railgun::kernel::CondLowering::CpSkip,
        });
        let err = lower_gfx11(&plan, rp::ArchLowering::for_arch("gfx1100"))
            .err()
            .expect("no COND_EXEC");
        assert!(err.contains("CP-skip"), "{err}");
    }

    #[test]
    fn gfx12_emission_is_unchanged_by_the_arch_neutral_loop() {
        let plan = plan_of(&[DEPENDENT, STRICT]);
        let lowering = rp::ArchLowering::for_arch("gfx1201");
        let make = || new_commands(Pm4Architecture::Gfx12);
        let commands = emit_segment_with(
            &plan,
            &plan.segments[0],
            lowering,
            &make,
            true,
            Gfx11EntryAcquirePolicy::System,
            |_, _| Ok(()),
        )
        .unwrap();
        // The same sequence straight from the gfx12 builder (no dispatches).
        let mut expected = Gfx12Pm4CommandBuffer::new_static_stateful();
        expected.acquire_system_gfx12();
        expected.wait_compute_idle();
        expected.acquire_inter_node_gfx12();
        expected.wait_compute_idle();
        expected.acquire_system_gfx12();
        expected.wait_compute_idle();
        expected.acquire_system_gfx12();
        assert_eq!(commands.gfx12_dwords().unwrap(), expected.dwords());
        assert!(commands.gfx11_dwords().is_none());
    }

    #[test]
    fn gfx11_stream_gate_names_what_it_refuses() {
        let reject = |dwords: &[u32]| check_gfx11_stream(dwords).unwrap_err();
        // gfx12 system acquire (GCR 0xc3b1) and same-agent acquire (0x10180).
        let mut g12 = Gfx12Pm4CommandBuffer::new();
        g12.acquire_system_gfx12();
        assert!(reject(g12.dwords()).contains("ACQUIRE_MEM"));
        let mut g12 = Gfx12Pm4CommandBuffer::new();
        g12.acquire_inter_node_gfx12();
        assert!(reject(g12.dwords()).contains("ACQUIRE_MEM"));
        // The gfx11 VMEM-only rung (GCR 0x300) is opt-in and not lowered.
        let mut vmem = Gfx10Pm4CommandBuffer::new();
        vmem.acquire_inter_node_vmem();
        assert!(reject(vmem.dwords()).contains("ACQUIRE_MEM"));
        // A gfx12 pacing NOP.
        let nop = [(3u32 << 30) | (0x10 << 8), 0];
        assert!(reject(&nop).contains("NOP"));
        // COND_EXEC.
        let mut cond = Gfx10Pm4CommandBuffer::new_stateful();
        let region = cond.begin_cond_exec(0x1000);
        cond.end_cond_exec(region).unwrap();
        assert!(reject(cond.dwords()).contains("COND_EXEC"));
        // The interleave experiment register.
        let mut interleave = Gfx10Pm4CommandBuffer::new()
            .with_dispatch_interleave(Some(Gfx11DispatchInterleave::Threads64));
        interleave
            .dispatch_image(
                &image(),
                LaunchGeometry::from_hip_workgroups([1, 1, 1], [64, 1, 1]).unwrap(),
                0,
                &[0, 0],
            )
            .unwrap();
        assert!(reject(interleave.dwords()).contains("COMPUTE_DISPATCH_INTERLEAVE"));
        // Fences are not part of an owned stream.
        let mut fence = Gfx10Pm4CommandBuffer::new();
        fence.dependency_fence(0x2000, 1);
        assert!(reject(fence.dwords()).contains("opcode"));
        // A truncated packet and a non-type-3 header.
        assert!(reject(&[(3u32 << 30) | (0x15 << 8) | (3 << 16), 1]).contains("overruns"));
        assert!(reject(&[0x8000_0000]).contains("type-3"));
        // Empty is a (useless but well-formed) stream.
        assert_eq!(check_gfx11_stream(&[]), Ok(Gfx11StreamCensus::default()));
    }

    #[test]
    fn dispatch_dimensions_are_workgroups_on_gfx11_and_workitems_on_gfx12() {
        let (grid, block) = ([5, 2, 1], [256, 1, 1]);
        assert_eq!(
            native_dimensions(Pm4Architecture::Gfx11, grid, block),
            Ok([5, 2, 1])
        );
        assert_eq!(
            native_dimensions(Pm4Architecture::Gfx12, grid, block),
            Ok([1280, 2, 1])
        );
        // u32 overflow is a gfx12 (work-item) concern only.
        let (big, block) = ([1 << 24, 1, 1], [256, 1, 1]);
        assert_eq!(
            native_dimensions(Pm4Architecture::Gfx12, big, block),
            Err(0)
        );
        assert_eq!(
            native_dimensions(Pm4Architecture::Gfx11, big, block),
            Ok(big)
        );
    }

    #[test]
    fn only_exact_gfx1100_by_pci_and_gfx12_are_owned_lowering_targets() {
        let pci = DeviceSelect::Pci("0000:66:00.0".parse().unwrap());
        let ordinal = DeviceSelect::Ordinal(0);
        for select in [pci, ordinal] {
            assert_eq!(
                device_refusal(Pm4Architecture::Gfx12, "gfx1201", select),
                None
            );
            assert_eq!(
                device_refusal(Pm4Architecture::Gfx12, "gfx1200", select),
                None
            );
        }
        assert_eq!(device_refusal(Pm4Architecture::Gfx11, "gfx1100", pci), None);
        let by_ordinal = device_refusal(Pm4Architecture::Gfx11, "gfx1100", ordinal).unwrap();
        assert!(by_ordinal.contains("PCI identity"), "{by_ordinal}");
        for name in [
            "gfx1151", "gfx1101", "gfx1102", "gfx1150", "gfx1152", "GFX1100",
        ] {
            for select in [pci, ordinal] {
                let reason = device_refusal(Pm4Architecture::Gfx11, name, select).unwrap();
                assert!(
                    reason.contains(name) && reason.contains("refused"),
                    "{reason}"
                );
            }
        }
        assert!(device_refusal(Pm4Architecture::Gfx10, "gfx1030", pci)
            .unwrap()
            .contains("gfx10"));
    }

    #[test]
    fn gfx12_only_commands_refuse_on_a_gfx11_stream() {
        let mut commands = new_commands(Pm4Architecture::Gfx11);
        assert!(commands
            .gfx12_system_acquire()
            .unwrap_err()
            .contains("legacy PM4"));
        assert!(commands.gfx12_dwords().is_none());
        // The architecture-neutral system acquire is ROCr's gfx11 packet.
        commands.system_acquire();
        let mut reference = Gfx10Pm4CommandBuffer::new();
        reference.acquire_system();
        assert_eq!(commands.gfx11_dwords().unwrap(), reference.dwords());
    }
}
