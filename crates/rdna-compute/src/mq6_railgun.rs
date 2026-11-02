// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Explicit native MQ6 railgun programs on exact gfx1151.
//!
//! This is an authored, fixed-binding operator program, not a recorder or a
//! replacement for model decode capture. Existing native, Redline and hipGraph
//! routes are unchanged. Replay joins the current HIP producer stream before
//! submitting the authored PM4 IB and synchronously joins its terminal signal
//! before a HIP consumer can be queued by the caller.
//!
//! Tensor views are NOT allocation owners. The unsafe prepare/replay contract
//! requires the caller to keep every original allocation alive and unmoved.
//! Unknown quiescence requires retaining them until GPU reset/process exit;
//! Drop quarantines the owned transport rather than freeing live GPU addresses.

use std::fmt;
use std::sync::Arc;

use railgun::kernel::{Arg, KernelProgram};
use railgun::mq6::{self, Mq6Facts, Mq6Op, Mq6Part, Mq6Span};
use railgun::plan::{
    self, CacheRow, CacheTable, Hazard, Pm4Op, Pm4Plan, RowPolicy, Rung, Visibility,
};
use railgun::Binding;
use railgun_corpus::{ArtifactKey, JitCorpus, Lookup};
use redline_dispatch::aql::{
    load_symbols, Executable, Gfx10Pm4CommandBuffer, GpuDevice, KernargBuffer, KernargPool, Kernel,
    LaunchGeometry, PciBusId, Quiescence, Runtime, SingleQueuePm4Ib,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::{DType, Gpu, GpuTensor};

/// One matrix of a typed invocation. Weight bytes use MQ6G256V2 wire format.
#[derive(Clone, Copy)]
pub struct Mq6TensorPart<'a> {
    pub weight: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub m: u32,
}

/// One fixed-shape single/x4 invocation. `None` x4 slots encode null/M=0.
#[derive(Clone, Copy)]
pub enum Mq6Invocation<'a> {
    Single {
        part: Mq6TensorPart<'a>,
        x: &'a GpuTensor,
        k: u32,
        batch: u32,
    },
    X4 {
        parts: [Option<Mq6TensorPart<'a>>; 4],
        x: &'a GpuTensor,
        k: u32,
        batch: u32,
    },
}

/// Failure with explicit retirement status. Unknown never licenses freeing
/// external allocations or reusing a poisoned prepared program.
#[derive(Debug)]
pub struct Mq6RailgunError {
    pub detail: String,
    pub quiescence: Quiescence,
}

impl fmt::Display for Mq6RailgunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MQ6 railgun: {} ({:?} quiescence)",
            self.detail, self.quiescence
        )
    }
}

impl std::error::Error for Mq6RailgunError {}

fn failure(error: impl fmt::Display) -> Mq6RailgunError {
    Mq6RailgunError {
        detail: error.to_string(),
        quiescence: Quiescence::Proven,
    }
}

fn require(ok: bool, detail: &str) -> Result<(), Mq6RailgunError> {
    if ok {
        Ok(())
    } else {
        Err(failure(detail))
    }
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn check_gpu(gpu: &Gpu) -> Result<PciBusId, Mq6RailgunError> {
    gpu.bind_thread().map_err(failure)?;
    require(
        gpu.arch == "gfx1151" && gpu.arch_caps.is_gfx1151(),
        "exact gfx1151 required",
    )?;
    require(
        !gpu.replay.is_recording() && !gpu.replay.is_g0_observing(),
        "explicit programs cannot participate in Redline recording/G0 observation",
    )?;
    require(
        !gpu.graphs.capture_mode,
        "explicit programs cannot participate in HIP graph capture",
    )?;
    if let Some(stream) = gpu.stream_ref() {
        require(
            !gpu.hip.stream_is_capturing(stream).map_err(failure)?,
            "active HIP stream is capturing",
        )?;
    }
    gpu.hip
        .device_pci_bus_id(gpu.device_id)
        .map_err(failure)?
        .parse::<PciBusId>()
        .map_err(failure)
}

struct View {
    // Non-owning aliases for revalidation, never passed to free_tensor.
    tensor: GpuTensor,
    allocation: Binding,
}

fn allocation(gpu: &Gpu, tensor: &GpuTensor) -> Result<Binding, Mq6RailgunError> {
    require(
        !tensor.buf.is_vmm_owner() && !tensor.buf.is_host_mapped(),
        "VMM/host-mapped owners need a separate committed-range proof",
    )?;
    let attrs = gpu
        .hip
        .pointer_get_attributes(&tensor.buf)
        .map_err(failure)?;
    require(
        attrs.mem_type == hip_bridge::MemoryType::Device as u32
            && attrs.device == gpu.device_id
            && attrs.is_managed == 0,
        "only ordinary same-device HIP device allocations/subviews are supported",
    )?;
    let (base, bytes) = gpu
        .hip
        .mem_get_address_range(tensor.buf.as_ptr())
        .map_err(failure)?;
    let binding = Binding {
        base: base as usize as u64,
        bytes: bytes as u64,
    };
    let start = tensor.buf.as_ptr() as usize as u64;
    let end = start
        .checked_add(tensor.buf.size() as u64)
        .ok_or_else(|| failure("view overflows"))?;
    let allocation_end = binding
        .base
        .checked_add(binding.bytes)
        .ok_or_else(|| failure("allocation overflows"))?;
    require(
        start >= binding.base && start < allocation_end && end <= allocation_end,
        "tensor view is outside its actual HIP allocation",
    )?;
    Ok(binding)
}

fn span(gpu: &Gpu, tensor: &GpuTensor, views: &mut Vec<View>) -> Result<Mq6Span, Mq6RailgunError> {
    let binding = allocation(gpu, tensor)?;
    views.push(View {
        tensor: tensor.shallow_clone(),
        allocation: binding,
    });
    Ok(Mq6Span {
        address: tensor.buf.as_ptr() as usize as u64,
        allocation: binding,
        bytes: tensor.buf.size() as u64,
    })
}

fn part(
    gpu: &Gpu,
    p: Mq6TensorPart<'_>,
    views: &mut Vec<View>,
) -> Result<Mq6Part, Mq6RailgunError> {
    require(p.output.dtype == DType::F32, "MQ6 output must be F32")?;
    Ok(Mq6Part {
        weight: span(gpu, p.weight, views)?,
        output: span(gpu, p.output, views)?,
        m: p.m,
    })
}

fn conservative_table() -> CacheTable {
    CacheTable { arch: "gfx1151".into(), rows: [Hazard::Raw, Hazard::War, Hazard::Waw, Hazard::WarPreWriter]
        .into_iter().map(|hazard| CacheRow { hazard,
            visibility: Visibility { wait_idle: true, acquire: Rung::System,
                                      pre_writer: hazard == Hazard::WarPreWriter },
            evidence: "Conservative per-program WaitIdle + System; no gfx1151 cache-row receipt or global admission",
            g4_receipt: None }).collect() }
}

// Declaration order matters: drop/quiesce the queue before releasing any of
// its pointees. Unknown quiescence retains this entire box instead.
struct Owned {
    graph: SingleQueuePm4Ib,
    _kernargs: Vec<KernargBuffer>,
    _kernels: Vec<Kernel>,
    _executable: Executable,
    _pool: KernargPool,
    _device: GpuDevice,
    _runtime: Runtime,
    views: Vec<View>,
}

/// Reusable authored program. Shapes and bindings are immutable: changed
/// allocations require a new prepare; changed contents at the same pointers
/// are read on every replay after the HIP producer join.
pub struct PreparedMq6Program {
    owned: Option<Box<Owned>>,
    program: KernelProgram,
    plan: Pm4Plan,
    table: CacheTable,
    hip_device: i32,
    pci: PciBusId,
    identity: Value,
    ib_sha256: String,
    kernarg_images: Vec<Vec<u8>>,
    replay_count: u64,
    poisoned: bool,
    quiescence: Quiescence,
    hip_handoff_failed: bool,
}

impl PreparedMq6Program {
    /// Author, plan and retain a straight-line native MQ6 program; no dispatch
    /// is submitted here. Only verified canonical corpus facts are accepted.
    ///
    /// # Safety
    /// Every tensor must name a live, same-device allocation in the stated
    /// format, backed by ordinary hipMalloc allocations (not borrowed VMM or
    /// host-mapped aliases). All ORIGINAL allocations (including owners behind borrowed
    /// views) must remain live and unmoved through this program's destruction.
    /// Retained shallow views are not owners. No incompatible concurrent host
    /// or GPU access is allowed. After Unknown quiescence, retain external
    /// allocations until GPU reset/process exit even after dropping this object.
    pub unsafe fn prepare(
        gpu: &Gpu,
        corpus: &JitCorpus,
        invocations: &[Mq6Invocation<'_>],
    ) -> Result<Self, Mq6RailgunError> {
        let pci = check_gpu(gpu)?;
        require(
            !invocations.is_empty(),
            "program must contain at least one invocation",
        )?;
        let bundle = crate::kernels::QWEN4_LANE_ROWS_PM_GFX1151;
        require(
            digest(bundle) == mq6::BUNDLE_SHA256,
            "embedded MQ6 bundle identity differs",
        )?;
        let hit = match corpus
            .lookup(&ArtifactKey {
                arch: "gfx1151",
                toolchain_pin: corpus.toolchain_pin(),
                artifact_sha256: mq6::BUNDLE_SHA256,
            })
            .map_err(failure)?
        {
            Lookup::Hit(hit) => hit,
            Lookup::Missing => {
                return Err(failure(
                    "canonical MQ6 receipt is absent from the verified corpus",
                ))
            }
        };
        let facts = Mq6Facts::from_hit(&hit).map_err(failure)?;
        let elf = crate::scratch::device_code_object(bundle, "gfx1151")
            .ok_or_else(|| failure("canonical gfx1151 device ELF cannot be extracted"))?;
        require(
            digest(elf) == hit.receipt.device_elf_sha256 && digest(elf) == mq6::DEVICE_ELF_SHA256,
            "loaded ELF is not the certified device object",
        )?;

        let mut views = Vec::new();
        let mut ops = Vec::with_capacity(invocations.len());
        for invocation in invocations {
            let (x, k, batch) = match *invocation {
                Mq6Invocation::Single { x, k, batch, .. }
                | Mq6Invocation::X4 { x, k, batch, .. } => (x, k, batch),
            };
            require(x.dtype == DType::F32, "MQ6 activation must be F32")?;
            let x = span(gpu, x, &mut views)?;
            ops.push(match *invocation {
                Mq6Invocation::Single { part: p, .. } => Mq6Op::Single {
                    part: part(gpu, p, &mut views)?,
                    x,
                    k,
                    batch,
                },
                Mq6Invocation::X4 { parts, .. } => {
                    let mut output = std::array::from_fn(|_| None);
                    for (index, p) in parts.into_iter().enumerate() {
                        output[index] = p.map(|p| part(gpu, p, &mut views)).transpose()?;
                    }
                    Mq6Op::X4 {
                        parts: output,
                        x,
                        k,
                        batch,
                    }
                }
            });
        }
        let program = mq6::author(&facts, &ops).map_err(failure)?;
        require(
            program.nodes.iter().all(|node| node.lowering.is_pm4()),
            "authored program contains an uncertified/non-PM4 node",
        )?;
        let table = conservative_table();
        let plan = plan::plan(&program, &table, RowPolicy::UnreceiptedSystem);
        require(
            plan.segments.len() == 1
                && plan.segments[0].first == 0
                && plan.segments[0].end == program.nodes.len(),
            "program does not form one complete PM4 segment",
        )?;

        let runtime = Runtime::initialize(load_symbols().map_err(failure)?).map_err(failure)?;
        let mut candidates = runtime
            .gpu_devices()
            .map_err(failure)?
            .into_iter()
            .filter(|device| device.pci_bus_id() == pci);
        let device = candidates
            .next()
            .ok_or_else(|| failure("HIP PCI device has no matching ROCr GPU"))?;
        require(
            candidates.next().is_none() && device.name() == "gfx1151",
            "ambiguous or wrong-architecture ROCr PCI match",
        )?;
        let executable = Executable::load(&device, Arc::<[u8]>::from(elf)).map_err(failure)?;
        let pool = KernargPool::discover(&device).map_err(failure)?;
        let words = program.authored_words();
        require(
            words.is_empty(),
            "MQ6 shapes must be constants, not dynamic words",
        )?;
        let mut kernels = Vec::new();
        let mut kernargs = Vec::new();
        let mut images = Vec::new();
        let mut loaded = Vec::new();
        for (index, node) in program.nodes.iter().enumerate() {
            let kernel = executable
                .kernel(&format!("{}.kd", node.symbol))
                .map_err(failure)?;
            let metadata = kernel.metadata();
            require(
                metadata.group_segment_size == 0
                    && metadata.private_segment_size == 0
                    && !metadata.dynamic_callstack
                    && node.block == [32, 1, 1]
                    && node.dynamic_lds == 0,
                "MQ6 loaded workgroup/LDS/scratch contract differs",
            )?;
            let image = program
                .kernarg_image(index, metadata.kernarg_segment_size as usize, &words)
                .map_err(failure)?;
            let mut buffer = pool.allocate_for(metadata).map_err(failure)?;
            buffer.write_exact(&image).map_err(failure)?;
            loaded.push(json!({"symbol":node.symbol,"explicit_bytes":node.explicit_bytes,
                "loader_segment_bytes":metadata.kernarg_segment_size,"kernel_object":metadata.kernel_object,
                "kernarg_address":buffer.address() as usize as u64}));
            images.push(image);
            kernels.push(kernel);
            kernargs.push(buffer);
        }
        let mut commands = Gfx10Pm4CommandBuffer::new_stateful();
        for op in plan.segment_ops(&plan.segments[0]) {
            match op {
                Pm4Op::Entry => commands.acquire_system(),
                Pm4Op::Boundary(visibility) => {
                    if visibility.wait_idle {
                        commands.wait_compute_idle();
                    }
                    match visibility.acquire {
                        Rung::None => {}
                        Rung::System => commands.acquire_system(),
                        Rung::InterNode => {
                            return Err(failure(
                                "unreceipted gfx1151 program cannot use a weaker cache rung",
                            ))
                        }
                    }
                }
                Pm4Op::Dispatch(index) => {
                    let node = &program.nodes[index];
                    let geometry =
                        LaunchGeometry::from_workgroups(node.grid, [32, 1, 1]).map_err(failure)?;
                    device.validate_geometry(geometry).map_err(failure)?;
                    commands
                        .dispatch(&kernels[index], geometry, 0, kernargs[index].address())
                        .map_err(failure)?;
                }
                Pm4Op::Exit => {
                    commands.wait_compute_idle();
                    commands.acquire_system();
                }
                Pm4Op::SkipBegin(_) | Pm4Op::SkipEnd => {
                    return Err(failure(
                        "explicit MQ6 programs do not support conditional PM4 operations",
                    ));
                }
            }
        }
        let ib_sha256 = digest(&commands.as_bytes());
        let graph = SingleQueuePm4Ib::create_gfx11(&device, &pool, &commands).map_err(failure)?;
        let identity = json!({"bundle_sha256":mq6::BUNDLE_SHA256,"device_elf_sha256":hit.receipt.device_elf_sha256,
            "receipt_sha256":hit.record.receipt_sha256,"loaded":loaded,
            "ib_dwords":commands.len_dwords(),"toolchain_pin":corpus.toolchain_pin()});
        Ok(Self {
            owned: Some(Box::new(Owned {
                graph,
                _kernargs: kernargs,
                _kernels: kernels,
                _executable: executable,
                _pool: pool,
                _device: device,
                _runtime: runtime,
                views,
            })),
            program,
            plan,
            table,
            hip_device: gpu.device_id,
            pci,
            identity,
            ib_sha256,
            kernarg_images: images,
            replay_count: 0,
            poisoned: false,
            quiescence: Quiescence::Proven,
            hip_handoff_failed: false,
        })
    }

    /// Join HIP producers, execute the same authored IB, then join the PM4
    /// terminal completion. Successful return permits a following HIP consumer.
    ///
    /// # Safety
    /// The prepare allocation/lifetime contract still holds. Any producer is
    /// ordered on this Gpu's current HIP stream; other-stream producers must
    /// already be joined there. No incompatible external access may race replay.
    /// Unknown failure forbids freeing any pointee until GPU reset/process exit.
    pub unsafe fn replay(&mut self, gpu: &Gpu) -> Result<(), Mq6RailgunError> {
        if self.poisoned {
            return Err(Mq6RailgunError {
                detail: "prepared program is poisoned; it cannot be replayed".into(),
                quiescence: self.quiescence,
            });
        }
        let pci = check_gpu(gpu)?;
        require(
            gpu.device_id == self.hip_device && pci == self.pci,
            "prepared program belongs to another HIP/ROCr device",
        )?;
        let owned = self
            .owned
            .as_mut()
            .ok_or_else(|| failure("transport is absent"))?;
        for view in &owned.views {
            require(
                allocation(gpu, &view.tensor)? == view.allocation,
                "allocation changed; prepare a new program",
            )?;
        }
        let next = self
            .replay_count
            .checked_add(1)
            .ok_or_else(|| failure("replay counter exhausted"))?;
        let synced = match gpu.stream_ref() {
            Some(stream) => gpu.hip.stream_synchronize(stream),
            None => gpu.hip.stream_synchronize_default(),
        };
        if let Err(error) = synced {
            self.poisoned = true;
            self.hip_handoff_failed = true;
            self.quiescence = Quiescence::Unknown;
            return Err(Mq6RailgunError {
                detail: format!("HIP producer handoff failed; no MQ6 submission: {error}"),
                quiescence: self.quiescence,
            });
        }
        // SAFETY: this object's code/kernargs/IB remain owned; external tensor
        // lifetime and non-racing access are forwarded from the caller.
        match unsafe { owned.graph.replay_and_wait_checked() } {
            Ok(()) => {
                self.replay_count = next;
                self.quiescence = Quiescence::Proven;
                Ok(())
            }
            Err((error, quiescence)) => {
                self.poisoned = true;
                self.quiescence = quiescence;
                Err(Mq6RailgunError {
                    detail: error.to_string(),
                    quiescence,
                })
            }
        }
    }

    /// Prove owned queue retirement after a failed replay. This cannot repair
    /// a failed HIP producer synchronization or re-enable a poisoned program.
    pub fn quiesce(&mut self) -> Result<(), Mq6RailgunError> {
        if self.hip_handoff_failed {
            return Err(Mq6RailgunError {
                detail: "HIP producer retirement is unproven; GPU reset/process exit required"
                    .into(),
                quiescence: Quiescence::Unknown,
            });
        }
        if let Some(owned) = self.owned.as_mut() {
            owned.graph.quiesce().map_err(|error| {
                self.quiescence = Quiescence::Unknown;
                self.poisoned = true;
                Mq6RailgunError {
                    detail: error.to_string(),
                    quiescence: Quiescence::Unknown,
                }
            })?;
        }
        self.quiescence = Quiescence::Proven;
        Ok(())
    }

    /// Identity/program evidence only; never reads weights or activations.
    pub fn report(&self) -> Value {
        let nodes: Vec<_> = self.program.nodes.iter().map(|node| {
            let args: Vec<_> = node.args.iter().map(|(offset,arg)| match arg {
                Arg::Const(bytes) => json!({"offset":offset,"kind":"constant","bytes":bytes}),
                Arg::Resource { resource, interior } => json!({"offset":offset,"kind":"resource","resource":resource.0,"interior":interior}),
                Arg::Word(word) => json!({"offset":offset,"kind":"word","word":word.0}),
            }).collect();
            let effects:Vec<_> = node.effects.iter().map(|effect| json!({"arg_offset":effect.arg_offset,
                "resource":effect.resource.0,"classes":effect.classes,"bound":"allocation_wide"})).collect();
            json!({"symbol":node.symbol,"kernel_ref":node.kref,"grid":node.grid,"block":node.block,
                "dynamic_lds":node.dynamic_lds,"explicit_bytes":node.explicit_bytes,"segment_bytes":node.segment_bytes,
                "args":args,"hidden":node.hidden,"effects":effects,"read_cache":node.read_cache,"lowering":node.lowering})
        }).collect();
        let resources: Vec<_> = self
            .program
            .resources
            .iter()
            .map(|binding| json!({"base":binding.base,"bytes":binding.bytes}))
            .collect();
        let mut report = self.identity.clone();
        let output = report.as_object_mut().expect("identity is an object");
        output.insert("executing_backend".into(), json!("railgun"));
        output.insert("replay_count".into(), json!(self.replay_count));
        output.insert(
            "successful_replay_observed".into(),
            json!(self.replay_count > 0),
        );
        output.insert(
            "execution_status".into(),
            json!(if self.replay_count > 0 {
                "completed_replays_observed"
            } else if self.poisoned {
                "failed_or_not_submitted"
            } else {
                "prepared_not_executed"
            }),
        );
        output.insert("ib_sha256".into(), json!(self.ib_sha256));
        output.insert(
            "kernarg_sha256".into(),
            json!(self
                .kernarg_images
                .iter()
                .map(|image| digest(image))
                .collect::<Vec<_>>()),
        );
        output.insert("kernarg_images".into(), json!(self.kernarg_images));
        output.insert("kernel_program".into(), json!({"arch":self.program.arch,"nodes":nodes,"resources":resources,"words":self.program.words}));
        output.insert("plan".into(), json!(self.plan));
        output.insert("cache_table".into(), json!(self.table));
        output.insert("hip_device".into(), json!(self.hip_device));
        output.insert("pci_bus_id".into(), json!(self.pci.to_string()));
        output.insert("quiescence".into(), json!(format!("{:?}", self.quiescence)));
        output.insert(
            "ready".into(),
            json!(self.owned.is_some() && !self.poisoned),
        );
        output.insert("poisoned".into(), json!(self.poisoned));
        report
    }
}

impl Drop for PreparedMq6Program {
    fn drop(&mut self) {
        if let Some(mut owned) = self.owned.take() {
            if self.hip_handoff_failed || owned.graph.quiesce().is_err() {
                eprintln!("MQ6 railgun: quiescence unknown; retaining transport/code/kernargs/IB. External allocations must remain live until GPU reset/process exit");
                std::mem::forget(owned);
            }
        }
    }
}
