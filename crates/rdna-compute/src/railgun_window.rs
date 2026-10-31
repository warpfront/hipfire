// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The GPU-facing half of the certified railgun DFlash window (design A
//! §2, §5, §6): everything `railgun::window::PreparedWindow` needs from
//! `rdna-compute` without exposing any Redline type.
//!
//! * [`WindowCorpus`] opens the JIT receipt corpus under the loaded
//!   toolchain pin, answers per-`(object, symbol)` fact lookups and admits a
//!   window manifest against the exact plan.
//! * [`Gpu::railgun_probe_allocation`] resolves a device pointer to its
//!   allocation, [`Gpu::railgun_transport_digests`] names the copy/fill
//!   objects the HIP loader holds, and [`Gpu::lower_railgun_window`] lowers a
//!   directly authored program through railgun's own gfx12 transport
//!   (`replay::railgun_lower`).
//! * [`RailgunWindowQueue`] implements `railgun::window::WindowQueue` over
//!   that lowering: it rewrites only the kernarg images and grids that
//!   changed, publishes the host writes and replays the retained indirect
//!   buffer once per submit, blocking until the queue is idle.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use railgun::kernel::{KernelProgram, NodeFacts};
use railgun::plan::Pm4Plan;
use railgun::window::{NodeImage, QueueFailure, Quiescence, ResolvedStep, WindowQueue};
use railgun::Binding;
use railgun_corpus::window::{ConditionalWitness, NodeRoute, RuntimeNode, WindowManifest};
use railgun_corpus::{sha256_hex, JitCorpus};

pub use crate::replay::railgun_lower::LowerTiming;
use crate::replay::railgun_lower::{LoweringContext, OwnedWindowLowering, SelfCheckFailure};
use crate::replay::railgun_shadow::{arch_lowering, lookup_facts, probe};
use crate::replay::ReplayQuiescence;
use crate::Gpu;

/// Prefix of the failure `RailgunWindowQueue::submit` returns (quiescence
/// `Proven`, nothing dispatched) when the pre-submit self-check of the lowered
/// state fails: the owner treats it as a refusal to the HIP route.
pub const LOWERING_SELF_CHECK: &str = "lowering self-check";

/// Refusals shown by [`WindowCorpus::admit_manifest`] before it summarises
/// the rest as a count.
const SHOWN_REFUSALS: usize = 8;

/// The JIT receipt corpus under the loaded toolchain: per-`(object, symbol)`
/// node facts and window-manifest admission, both keyed by the arch and
/// toolchain pin of the [`Gpu`] it was opened for.
pub struct WindowCorpus {
    corpus: JitCorpus,
    arch: String,
    pin: String,
    facts: HashMap<(String, String), Result<Arc<NodeFacts>, String>>,
    /// The manifests of the last witnessed admission, parsed once per path.
    manifests: Option<(PathBuf, Vec<(PathBuf, WindowManifest)>)>,
}

impl WindowCorpus {
    /// Open the corpus at `dir` for `gpu`'s arch. Refused when the corpus was
    /// not built under the toolchain `gpu` compiles with: its pin
    /// (`SHA-256(toolchain_id)`, as the M1 shadow derives it) must equal the
    /// corpus pin.
    pub fn open(gpu: &Gpu, dir: &Path) -> Result<Self, String> {
        let pin = sha256_hex(gpu.compiler.toolchain_id().as_bytes());
        Self::open_pinned(gpu.arch.clone(), pin, dir)
    }

    fn open_pinned(arch: String, pin: String, dir: &Path) -> Result<Self, String> {
        let corpus =
            JitCorpus::open(dir).map_err(|e| format!("railgun JIT corpus rejected: {e}"))?;
        if pin != corpus.toolchain_pin() {
            return Err(format!(
                "toolchain pin {} is not the corpus pin {}",
                &pin[..16],
                &corpus.toolchain_pin()[..16]
            ));
        }
        Ok(Self {
            corpus,
            arch,
            pin,
            facts: HashMap::new(),
            manifests: None,
        })
    }

    /// The arch every lookup is made under.
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// SHA-256 of the toolchain id every lookup is made under.
    pub fn toolchain_pin(&self) -> &str {
        &self.pin
    }

    /// The facts of `symbol` in the object whose SHA-256 is
    /// `artifact_sha256_hex`. Hits and refusals are cached per
    /// `(object, symbol)`; the refusal text is the M1 shadow's.
    pub fn facts(
        &mut self,
        artifact_sha256_hex: &str,
        symbol: &str,
    ) -> Result<Arc<NodeFacts>, String> {
        let key = (artifact_sha256_hex.to_owned(), symbol.to_owned());
        if let Some(hit) = self.facts.get(&key) {
            return hit.clone();
        }
        let result = lookup_facts(
            &self.corpus,
            &self.arch,
            &self.pin,
            artifact_sha256_hex,
            symbol,
        );
        self.facts.insert(key, result.clone());
        result
    }

    /// Admit the plan `nodes` (`(symbol, artifact_sha256_hex)` in plan order,
    /// every node a railgun node) against the window manifest at
    /// `manifest_json_path`. Every refusal is reported: the first
    /// [`SHOWN_REFUSALS`] verbatim and the count of the rest.
    pub fn admit_manifest(
        &self,
        manifest_json_path: &Path,
        nodes: &[(String, String)],
    ) -> Result<(), String> {
        let manifests = read_manifests(manifest_json_path)?;
        let manifest = select_manifest(&manifests, nodes)?;
        let plan = runtime_nodes(nodes);
        manifest
            .admit(&self.arch, &self.pin, &plan)
            .map_err(summarise_refusals)
    }

    /// Like [`Self::admit_manifest`], for a plan with `conditional` nodes: each
    /// node's guarded case needs the matching bind witness. `cases` are
    /// `(node index, case digest, witness kind)` exactly as
    /// `IndexedBindWitness::discharged_cases` reports them, i.e. discharged by
    /// this window's runtime bind, never taken from the manifest or receipts.
    /// The manifest(s) are parsed once per path and kept; see
    /// [`read_manifests`] for the directory form and [`select_manifest`] for
    /// how one is chosen per window shape.
    pub fn admit_manifest_witnessed(
        &mut self,
        manifest_json_path: &Path,
        nodes: &[(String, String)],
        cases: &[(usize, &str, &str)],
    ) -> Result<(), String> {
        if self.manifests.as_ref().map(|(p, _)| p.as_path()) != Some(manifest_json_path) {
            let manifests = read_manifests(manifest_json_path)?;
            self.manifests = Some((manifest_json_path.to_owned(), manifests));
        }
        let manifest = select_manifest(&self.manifests.as_ref().expect("just stored").1, nodes)?;
        let plan = runtime_nodes(nodes);
        let witnesses: Vec<ConditionalWitness<'_>> = cases
            .iter()
            .map(|(index, case, kind)| ConditionalWitness {
                index: *index as u32,
                case_sha256: case,
                kind,
            })
            .collect();
        manifest
            .admit_with(&self.arch, &self.pin, &plan, &witnesses)
            .map_err(summarise_refusals)
    }
}

/// Print where every mismatching kernarg buffer sits relative to the bound
/// allocations and to the other kernarg buffers, and which bound image words
/// point into (or just beside) it: the evidence for telling VA aliasing, an
/// out-of-bounds write from a neighbouring allocation and a stale cache line
/// apart.
fn mismatch_report(
    f: &SelfCheckFailure,
    resources: &[(u64, u64)],
    names: &[String],
    kernargs: &[(u64, usize, usize)],
    images: &[&[u8]],
) -> Vec<String> {
    let mut out = Vec::new();
    const NEAR: u64 = 1 << 16;
    let label = |r: usize| match names.get(r).map(String::as_str) {
        Some(n) if !n.is_empty() => format!("r{r}({n})"),
        _ => format!("r{r}"),
    };
    let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
    out.push(format!(
        "[dflash-railgun]   {} of {} kernarg buffers differ; pool VA span {:#x}..{:#x}",
        f.kernarg.len(),
        kernargs.len(),
        kernargs.iter().map(|k| k.0).min().unwrap_or(0),
        kernargs.iter().map(|k| k.0 + k.2 as u64).max().unwrap_or(0)
    ));
    for m in f.kernarg.iter().take(8) {
        let (lo, hi) = (m.address, m.address + m.allocation.max(m.length) as u64);
        let overlapping: Vec<String> = resources
            .iter()
            .enumerate()
            .filter(|(_, (b, n))| *b < hi && lo < b + n)
            .map(|(r, (b, n))| format!("{}[{b:#x}+{n}]", label(r)))
            .collect();
        let below = resources
            .iter()
            .enumerate()
            .filter(|(_, (b, n))| b + n <= lo)
            .max_by_key(|(_, (b, n))| b + n)
            .map(|(r, (b, n))| format!("{} ends {} B below", label(r), lo - (b + n)));
        let above = resources
            .iter()
            .enumerate()
            .filter(|(_, (b, _))| *b >= hi)
            .min_by_key(|(_, (b, _))| *b)
            .map(|(r, (b, _))| format!("{} starts {} B above", label(r), b - hi));
        let neighbours: Vec<usize> = kernargs
            .iter()
            .enumerate()
            .filter(|(i, k)| *i != m.node && k.0 < hi + NEAR && lo < k.0 + k.2 as u64 + NEAR)
            .map(|(i, _)| i)
            .collect();
        let pointing: Vec<String> = images
            .iter()
            .enumerate()
            .flat_map(|(n, img)| {
                img.chunks_exact(8).enumerate().filter_map(move |(w, c)| {
                    let v = u64::from_le_bytes(c.try_into().expect("8 bytes"));
                    (v + NEAR >= lo && v < hi + NEAR).then(|| format!("node {n} +{}={v:#x}", w * 8))
                })
            })
            .take(8)
            .collect();
        out.push(format!(
            "[dflash-railgun]   node {} {}: buffer {:#x}+{} (alloc {}), {} differing bytes; \
             expected {} actual {}; overlaps program resources: {}; nearest: {} / {}; \
             kernarg buffers within 64 KiB: {neighbours:?}; bound image words near it: {pointing:?}",
            m.node,
            m.symbol,
            m.address,
            m.length,
            m.allocation,
            m.differing_bytes,
            hex(&m.expected_head),
            hex(&m.actual_head),
            if overlapping.is_empty() { "none".to_owned() } else { overlapping.join(" ") },
            below.unwrap_or_else(|| "-".into()),
            above.unwrap_or_else(|| "-".into()),
        ));
    }
    out
}

fn read_manifest(path: &Path) -> Result<WindowManifest, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("window manifest {}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("window manifest {}: {e}", path.display()))
}

/// The manifest(s) at `path`: one file, or every `*.json` of a directory
/// (sorted by name), each certifying one window node sequence. A window's
/// node sequence depends on its shape family (ring wrap, concat segments,
/// attention block), so one manifest covers one family; a directory of
/// manifests covers several.
fn read_manifests(path: &Path) -> Result<Vec<(PathBuf, WindowManifest)>, String> {
    if !path.is_dir() {
        return Ok(vec![(path.to_owned(), read_manifest(path)?)]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|e| format!("window manifest directory {}: {e}", path.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!(
            "window manifest directory {} holds no *.json manifest",
            path.display()
        ));
    }
    files
        .into_iter()
        .map(|f| read_manifest(&f).map(|m| (f, m)))
        .collect()
}

/// The manifest that certifies `nodes` (`(symbol, artifact_sha256_hex)` in
/// plan order). With a single manifest it is that one, so a mismatch is
/// reported node by node by the admission itself. With several, it is the
/// first whose node sequence equals the plan's exactly; none is a refusal that
/// names the plan's sequence so the missing manifest can be generated. Pure
/// host work: nothing is mutated.
fn select_manifest<'m>(
    manifests: &'m [(PathBuf, WindowManifest)],
    nodes: &[(String, String)],
) -> Result<&'m WindowManifest, String> {
    if let [(_, only)] = manifests {
        return Ok(only);
    }
    let same = |m: &WindowManifest| {
        m.nodes.len() == nodes.len()
            && m.nodes
                .iter()
                .zip(nodes)
                .all(|(r, (symbol, sha))| &r.symbol == symbol && &r.artifact_sha256 == sha)
    };
    manifests
        .iter()
        .map(|(_, m)| m)
        .find(|m| same(m))
        .ok_or_else(|| {
            let sequence: String = nodes.iter().map(|(s, a)| format!("{s}\t{a}\n")).collect();
            format!(
                "no window manifest of the {} loaded covers this window's node sequence \
                 ({} nodes, sequence {}); dump it with HIPFIRE_DFLASH_RAILGUN_SEQUENCE_OUT \
                 and generate its manifest",
                manifests.len(),
                nodes.len(),
                &sha256_hex(sequence.as_bytes())[..16]
            )
        })
}

fn runtime_nodes(nodes: &[(String, String)]) -> Vec<RuntimeNode<'_>> {
    nodes
        .iter()
        .map(|(symbol, sha)| RuntimeNode {
            symbol,
            artifact_sha256: sha,
            route: NodeRoute::Railgun,
        })
        .collect()
}

fn summarise_refusals(refusals: Vec<railgun_corpus::window::WindowRefusal>) -> String {
    let shown: Vec<String> = refusals
        .iter()
        .take(SHOWN_REFUSALS)
        .map(ToString::to_string)
        .collect();
    let rest = refusals.len().saturating_sub(SHOWN_REFUSALS);
    let more = if rest == 0 {
        String::new()
    } else {
        format!("; and {rest} more")
    };
    format!(
        "window manifest refused the plan ({} refusals): {}{more}",
        refusals.len(),
        shown.join("; ")
    )
}

/// SHA-256 object digests of railgun's typed copy and fill kernels as the
/// HIP loader holds them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RailgunTransportDigests {
    pub copy: [u8; 32],
    pub fill: [u8; 32],
}

impl Gpu {
    /// The allocation containing device address `addr` (`hipMemGetAddressRange`).
    pub fn railgun_probe_allocation(&self, addr: u64) -> Result<Binding, String> {
        probe(&self.hip, addr)
    }

    /// Load railgun's copy and fill kernels (idempotent) and return the
    /// digests of the code objects that own them. Refused inside plan mode:
    /// a plan-mode load would be recorded, not performed.
    pub fn railgun_transport_digests(&mut self) -> Result<RailgunTransportDigests, String> {
        if self.plan_mode_active() {
            return Err("railgun transport kernels cannot be loaded inside plan mode".to_owned());
        }
        self.ensure_kernel_public(
            railgun::copy::MODULE,
            railgun::copy::SOURCE,
            railgun::copy::COPY_SYMBOL,
        )
        .map_err(|e| format!("{}: {e}", railgun::copy::COPY_SYMBOL))?;
        self.ensure_kernel_public(
            railgun::copy::FILL_MODULE,
            railgun::copy::FILL_SOURCE,
            railgun::copy::FILL_SYMBOL,
        )
        .map_err(|e| format!("{}: {e}", railgun::copy::FILL_SYMBOL))?;
        let digest = |symbol: &str| -> Result<[u8; 32], String> {
            self.loaded_owner(symbol)
                .map(|(_, artifact)| artifact.id().0)
                .map_err(|e| format!("{symbol}: {e}"))
        };
        Ok(RailgunTransportDigests {
            copy: digest(railgun::copy::COPY_SYMBOL)?,
            fill: digest(railgun::copy::FILL_SYMBOL)?,
        })
    }

    /// The shared lowering context of this device: one HSA runtime and device,
    /// the host and retained kernarg pools and every loaded code object,
    /// created once by the owner and handed to every
    /// [`Self::lower_railgun_window`]. gfx12 only.
    pub fn railgun_lowering_context(&self) -> Result<RailgunLoweringContext, String> {
        let ordinal = usize::try_from(self.device_id)
            .map_err(|_| format!("device ordinal {} is negative", self.device_id))?;
        Ok(RailgunLoweringContext {
            inner: LoweringContext::new(ordinal)?,
        })
    }

    /// `(name, base, bytes)` of every live `Gpu` scratch slot (the lazily
    /// grown, replaced-on-growth buffers of `scratch::ScratchState`). A cached
    /// railgun skeleton that binds a pointer into one is stale after that slot
    /// grows (see `scratch::scratch_growth_epoch`); this names them for the
    /// `HIPFIRE_DFLASH_RAILGUN_DEBUG` resource table.
    pub fn railgun_scratch_slots(&self) -> Vec<(String, u64, u64)> {
        let sc = &self.scratch;
        let mut out = Vec::new();
        macro_rules! buf {
            ($($f:ident),* $(,)?) => {$(
                if let Some(b) = &sc.$f {
                    out.push((stringify!($f).to_owned(), b.as_ptr() as usize as u64, b.size() as u64));
                }
            )*};
        }
        macro_rules! tensor {
            ($($f:ident),* $(,)?) => {$(
                if let Some(t) = &sc.$f {
                    out.push((stringify!($f).to_owned(), t.buf.as_ptr() as usize as u64, t.buf.size() as u64));
                }
            )*};
        }
        tensor!(
            mq_signs1,
            mq_signs2,
            mq_signs1_128,
            mq_signs2_128,
            mq_x_rot,
            argmax_host,
            gemv_residual_tmp,
            mq6_fp8_x_rot,
            paro_x_scratch
        );
        buf!(
            mq_x_rot_fp8,
            mq_x_q8,
            mq_x_scales,
            mq_rmsnorm_wavegrid_scratch,
            gdn_pair_counters,
            mq_rmsnorm_awq_wavegrid_scratch,
            fp16_x_scratch,
            qsa_select_scores,
            gdn_state_f32,
            qsa_gather_f16,
            fp8_x_scratch,
            q8_1_mmq_x_scratch,
            int4_mmq_x_scratch,
            int4_mmq_down_scratch,
            q8_a8_x_scratch,
            int8_mmq_x_scratch,
            mq4v2_fp8_x_scratch,
            mq4v2_fp8_half_sums_scratch,
            mq4v2_fp8_row_scales_scratch,
            fp8_verify_x8_scratch,
            fp8_verify_scales_scratch,
            fp8_f2_weights,
            ksplit_det_partials,
            sample_partials,
            fa2_q16_scratch,
            fa2_fp8_q_scratch
        );
        out
    }

    /// What the last failed pre-submit self-check found about the VA ranges of
    /// its mismatching kernarg buffers that only the HIP runtime can answer:
    /// whether `hipMemGetAddressRange` tracks an allocation containing each
    /// one. Consumes the record.
    pub fn railgun_probe_self_check_failure(&self) -> Option<String> {
        let ranges = LAST_SELF_CHECK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()?;
        let lines: Vec<String> = ranges
            .iter()
            .map(|(node, addr, len)| match probe(&self.hip, *addr) {
                Ok(b) => format!(
                    "node {node}: kernarg {addr:#x}+{len} lies INSIDE HIP-tracked allocation {:#x}+{} (VA aliasing)",
                    b.base, b.bytes
                ),
                Err(e) => format!("node {node}: kernarg {addr:#x}+{len} is not in a HIP-tracked allocation ({e})"),
            })
            .collect();
        Some(lines.join("; "))
    }

    /// Lower `program` (one single-segment, all-PM4 window) through railgun's
    /// own gfx12 transport on this device. Every node's symbol must be loaded
    /// and owned by exactly the code object its `kref` certifies; a node
    /// without a `kref`, or whose loaded object digest differs, refuses the
    /// whole lowering. Nothing is submitted.
    pub fn lower_railgun_window(
        &mut self,
        ctx: &mut RailgunLoweringContext,
        program: &KernelProgram,
        plan: &Pm4Plan,
    ) -> Result<RailgunWindowQueue, String> {
        let artifacts = self.railgun_resolve_artifacts(program)?;
        ctx.lower(&self.arch, program, plan, &artifacts)
    }

    /// The loaded code object behind every node of `program`, each proven to
    /// be exactly the object the node's `kref` certifies (a node without a
    /// `kref`, or whose loaded digest differs, refuses). Host-side and
    /// quick; the result is what [`RailgunLoweringContext::lower`] needs and
    /// is `Send`, so the lowering itself can run on another thread.
    pub fn railgun_resolve_artifacts(
        &self,
        program: &KernelProgram,
    ) -> Result<RailgunArtifacts, String> {
        if program.arch != self.arch {
            return Err(format!(
                "program arch {} is not the device arch {}",
                program.arch, self.arch
            ));
        }
        let mut artifacts = Vec::with_capacity(program.nodes.len());
        for (i, node) in program.nodes.iter().enumerate() {
            let kref = node.kref.as_ref().ok_or_else(|| {
                format!("node {i} {}: no certified kernel reference", node.symbol)
            })?;
            let (_, artifact) = self
                .loaded_owner(&node.symbol)
                .map_err(|e| format!("node {i} {}: {e}", node.symbol))?;
            let loaded = artifact.id().to_hex();
            if loaded != kref.artifact_sha256 {
                return Err(format!(
                    "node {i} {}: loaded object {loaded} is not the certified object {}",
                    node.symbol, kref.artifact_sha256
                ));
            }
            artifacts.push(artifact.clone());
        }
        Ok(RailgunArtifacts {
            arch: self.arch.clone(),
            artifacts,
        })
    }
}

/// The shared gfx12 lowering context of one owner (see
/// [`Gpu::railgun_lowering_context`]): created once, lent to every skeleton
/// lowering. Dropping it releases only what no skeleton still holds.
pub struct RailgunLoweringContext {
    inner: LoweringContext,
}

impl RailgunLoweringContext {
    /// Distinct code objects loaded so far (each once for the whole context).
    pub fn loaded_objects(&self) -> usize {
        self.inner.loaded_objects()
    }

    /// Lower `program` (one single-segment, all-PM4 window) through railgun's
    /// own gfx12 transport with the code objects `artifacts` resolved on the
    /// owner thread. Needs no `Gpu`, so it may run on a background thread.
    /// Nothing is submitted.
    pub fn lower(
        &mut self,
        arch: &str,
        program: &KernelProgram,
        plan: &Pm4Plan,
        artifacts: &RailgunArtifacts,
    ) -> Result<RailgunWindowQueue, String> {
        if artifacts.arch != arch || program.arch != arch {
            return Err(format!(
                "program arch {}, artifacts for {}, device arch {arch}",
                program.arch, artifacts.arch
            ));
        }
        let (lowering, timing) =
            self.inner
                .lower(program, plan, arch_lowering(arch), &artifacts.artifacts)?;
        Ok(RailgunWindowQueue {
            resources: program
                .resources
                .iter()
                .map(|r| (r.base, r.bytes))
                .collect(),
            resource_names: Vec::new(),
            symbols: program.nodes.iter().map(|n| n.symbol.clone()).collect(),
            self_check: false,
            last: vec![None; program.nodes.len()],
            grids: program.nodes.iter().map(|n| Some(n.grid)).collect(),
            lowering,
            failure: None,
            timing,
        })
    }
}

/// The loaded code objects of one program's nodes, resolved by
/// [`Gpu::railgun_resolve_artifacts`].
#[derive(Clone)]
pub struct RailgunArtifacts {
    arch: String,
    artifacts: Vec<crate::code_object::RecordedArtifact>,
}

// SAFETY: the context is owned by one window builder and only used through
// `&mut self` on the thread that holds the `Gpu`, exactly like
// `RailgunWindowQueue` (see the justification there): its HSA runtime, pools
// and loaded objects are plain handles with no thread affinity.
unsafe impl Send for RailgunLoweringContext {}

/// Failing kernarg buffers of the last self-check: `(node, VA, length)`.
static LAST_SELF_CHECK: Mutex<Option<Vec<(usize, u64, usize)>>> = Mutex::new(None);

/// `railgun::window::WindowQueue` over railgun's owned gfx12 lowering: one
/// retained indirect buffer, one kernarg buffer per node. Only the kernarg
/// images and grids that differ from what the queue last held are rewritten;
/// each submit replays the whole tape and blocks until the queue is idle, so
/// a successful submit has already proven quiescence.
pub struct RailgunWindowQueue {
    lowering: OwnedWindowLowering,
    /// Per node: its symbol (self-check messages).
    symbols: Vec<String>,
    /// `(base, bytes)` of every program resource (self-check neighbourhood).
    resources: Vec<(u64, u64)>,
    /// Per program resource: the owner tensor or `Gpu` scratch slot it is
    /// (`""` when unknown); set by the builder under the debug arm.
    resource_names: Vec<String>,
    /// Verify the lowered state against the bound images before every replay.
    self_check: bool,
    /// Per node: the kernarg image last written, `None` while unknown (never
    /// written, or a write failed part-way).
    last: Vec<Option<Vec<u8>>>,
    /// Per node: the grid the indirect buffer currently dispatches, `None`
    /// while unknown (a patch failed). Initially the authored grids.
    grids: Vec<Option<[u32; 3]>>,
    /// The outcome of the last submit when it failed; what `wait` reports.
    failure: Option<QueueFailure>,
    /// Where lowering this queue spent its time.
    timing: LowerTiming,
}

// SAFETY: `RailgunWindowQueue` is only ever driven through `&mut self`, so
// one thread uses it at a time; moving it between threads is the only thing
// `WindowQueue: Send` asks for. What blocks the auto impl is raw handles
// (`NonNull` queue/kernarg pointers in `AqlQueue`/`KernargBuffer`) into HSA
// objects this value exclusively owns. Checked in `redline-rocr` and
// `redline-dispatch::aql` (the types `SingleQueuePm4Ib`, `Kernel` and
// `KernargBuffer` are made of): no `thread_local!`, no `ThreadId`/
// `thread::current`, no `Rc`/`RefCell`/`Cell` (the `Rc` in
// `redline-dispatch::hip` belongs to the HIP plan recorder, not these
// types); shared state is `Arc` over `Send + Sync` data (the symbol table's
// keepalive is `Arc<dyn Send + Sync>`) and the queue-fault word the HSA error
// callback writes is an atomic. The HSA runtime API (queue doorbells,
// signals, memory pools) is callable from any thread, so nothing here is
// thread-affine.
unsafe impl Send for RailgunWindowQueue {}

impl RailgunWindowQueue {
    /// Command dwords of the retained indirect buffer.
    pub fn command_dwords(&self) -> u32 {
        self.lowering.command_dwords()
    }

    /// Where lowering this queue spent its time (code objects loaded vs
    /// kernarg allocation / IB emission / graph creation).
    pub fn lowering_timing(&self) -> LowerTiming {
        self.timing
    }

    /// Name the program's resources (index order) for the self-check report.
    pub fn set_resource_names(&mut self, names: Vec<String>) {
        self.resource_names = names;
    }

    /// Arm the pre-submit self-check (`HIPFIRE_DFLASH_RAILGUN_DEBUG`): before
    /// each replay, read every kernarg buffer and every dispatch packet of the
    /// lowering back and compare it with the bound images and grids; a mismatch
    /// refuses the submit (nothing dispatched) after printing the first one.
    pub fn set_self_check(&mut self, on: bool) {
        self.self_check = on;
    }

    /// Number of lowered nodes; `submit` takes exactly this many images.
    pub fn node_count(&self) -> usize {
        self.last.len()
    }

    fn proven(message: String) -> QueueFailure {
        QueueFailure {
            message,
            quiescence: Quiescence::Proven,
        }
    }

    /// Write the images and grids that changed; every failure here happens
    /// before any dispatch, so the queue is still quiescent.
    fn prepare(&mut self, nodes: &[NodeImage<'_>]) -> Result<(), QueueFailure> {
        let mut wrote = false;
        for (i, image) in nodes.iter().enumerate() {
            if self.last[i].as_deref() != Some(image.kernarg) {
                // A failed write may have landed part of the image.
                let previous = self.last[i].take();
                self.lowering
                    .write_kernarg(i, image.kernarg)
                    .map_err(Self::proven)?;
                let mut kept = previous.unwrap_or_default();
                kept.clear();
                kept.extend_from_slice(image.kernarg);
                self.last[i] = Some(kept);
                wrote = true;
            }
            if self.grids[i] != Some(image.grid) {
                self.grids[i] = None;
                self.lowering
                    .patch_grid(i, image.grid)
                    .map_err(Self::proven)?;
                self.grids[i] = Some(image.grid);
            }
        }
        if wrote {
            self.lowering.publish_host_writes();
        }
        Ok(())
    }
}

impl WindowQueue for RailgunWindowQueue {
    fn submit(
        &mut self,
        nodes: &[NodeImage<'_>],
        steps: &[ResolvedStep],
    ) -> Result<(), QueueFailure> {
        if let Some(failure) = &self.failure {
            if failure.quiescence == Quiescence::Unknown {
                return Err(failure.clone());
            }
        }
        self.failure = None;
        if !steps.is_empty() {
            let failure = Self::proven(format!(
                "{} typed steps submitted: copy and fill are program nodes on the railgun route",
                steps.len()
            ));
            self.failure = Some(failure.clone());
            return Err(failure);
        }
        if nodes.len() != self.last.len() {
            let failure = Self::proven(format!(
                "{} node images for {} lowered nodes",
                nodes.len(),
                self.last.len()
            ));
            self.failure = Some(failure.clone());
            return Err(failure);
        }
        if let Err(failure) = self.prepare(nodes) {
            self.failure = Some(failure.clone());
            return Err(failure);
        }
        if self.self_check {
            let images: Vec<&[u8]> = nodes.iter().map(|n| n.kernarg).collect();
            let grids: Vec<[u32; 3]> = nodes.iter().map(|n| n.grid).collect();
            if let Err(f) = self.lowering.verify_state(&self.symbols, &images, &grids) {
                eprintln!("[dflash-railgun] {LOWERING_SELF_CHECK} FAILED: {}", f.first);
                for line in mismatch_report(
                    &f,
                    &self.resources,
                    &self.resource_names,
                    &self.lowering.kernarg_ranges(),
                    &images,
                ) {
                    eprintln!("{line}");
                }
                *LAST_SELF_CHECK.lock().unwrap_or_else(|e| e.into_inner()) = Some(
                    f.kernarg
                        .iter()
                        .take(8)
                        .map(|m| (m.node, m.address, m.allocation.max(m.length)))
                        .collect(),
                );
                // Nothing was dispatched: the queue is quiescent. Distrust the
                // cached image/grid state so the next submit rewrites it all.
                self.last.iter_mut().for_each(|l| *l = None);
                self.grids.iter_mut().for_each(|g| *g = None);
                let failure = Self::proven(format!("{LOWERING_SELF_CHECK}: {}", f.first));
                self.failure = Some(failure.clone());
                return Err(failure);
            }
            eprintln!(
                "[dflash-railgun] {LOWERING_SELF_CHECK} ok: {} kernarg buffers and {} dispatches match",
                nodes.len(),
                nodes.len()
            );
        }
        // SAFETY: every pointer in the kernarg images just written was bound
        // by the window binding key (`WindowBindingKey`), which proves each
        // allocation it names is live and unmoved for this generation; the
        // lowering owns its kernargs, kernels and indirect buffer, and the
        // queue is quiescent (the previous replay waited, or none ran).
        let replayed = unsafe { self.lowering.replay_and_wait() };
        replayed.map_err(|failure| {
            let failure = QueueFailure {
                message: failure.error,
                quiescence: match failure.quiescence {
                    ReplayQuiescence::Proven => Quiescence::Proven,
                    ReplayQuiescence::Unknown => Quiescence::Unknown,
                },
            };
            self.failure = Some(failure.clone());
            failure
        })
    }

    fn wait(&mut self) -> Result<(), QueueFailure> {
        // `submit` blocks until the queue is idle, so a successful submit has
        // already proven quiescence and there is nothing left to wait for.
        match &self.failure {
            Some(failure) => Err(failure.clone()),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::railgun_lower::KernargMismatch;
    use railgun_corpus::window::{
        sequence_digest, WindowNodeRecord, WHOLE16_SHAPE, WINDOW_MANIFEST_SCHEMA,
        WINDOW_MANIFEST_VERSION,
    };
    use railgun_corpus::{
        BuildHost, CorpusIndex, ToolchainPin, CORPUS_INDEX, CORPUS_SCHEMA, CORPUS_VERSION,
    };
    use std::path::PathBuf;

    const ARCH: &str = "gfx1201";
    const TOOLCHAIN_ID: &str = "AMD clang version test";

    fn sha(c: char) -> String {
        std::iter::repeat(c).take(64).collect()
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("rdna-railgun-window-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn write_corpus(dir: &Path, pin_of: &str) {
        let index = CorpusIndex {
            schema: CORPUS_SCHEMA.to_owned(),
            version: CORPUS_VERSION,
            source_commit: "0".repeat(40),
            build_host: BuildHost {
                key: sha('1'),
                hostname: "test".into(),
                hipcc: "hipcc".into(),
                rocm_env_root: None,
                clang_sha256: sha('2'),
                header_set_sha256: sha('3'),
                build_dir: "/tmp/build".into(),
            },
            toolchain: ToolchainPin {
                pin: sha256_hex(pin_of.as_bytes()),
                id: pin_of.to_owned(),
                version_sha256: sha('4'),
            },
            records: Vec::new(),
            rejected: Vec::new(),
            programs: Vec::new(),
        };
        std::fs::write(
            dir.join(CORPUS_INDEX),
            serde_json::to_vec(&index).expect("index serialises"),
        )
        .expect("write index");
    }

    fn record(index: u32, symbol: &str, artifact: &str) -> WindowNodeRecord {
        WindowNodeRecord {
            index,
            phase: "verify_body".into(),
            class: String::new(),
            symbol: symbol.into(),
            artifact_sha256: artifact.into(),
            device_elf_sha256: sha('d'),
            text_sha256: sha('e'),
            kd_sha256: sha('f'),
            metadata_sha256: sha('a'),
            receipt_sha256: sha('b'),
            a1_status: "proven".into(),
            case_sha256: String::new(),
            required_witnesses: Vec::new(),
            kernarg_size: 16,
            explicit_end: 16,
            hidden_loaded: Vec::new(),
        }
    }

    fn write_manifest(dir: &Path, nodes: Vec<WindowNodeRecord>) -> PathBuf {
        let manifest = WindowManifest {
            schema: WINDOW_MANIFEST_SCHEMA.into(),
            version: WINDOW_MANIFEST_VERSION,
            shape: WHOLE16_SHAPE.into(),
            arch: ARCH.into(),
            toolchain_pin: sha256_hex(TOOLCHAIN_ID.as_bytes()),
            sequence_sha256: sequence_digest(&nodes),
            nodes,
            abi: Vec::new(),
        };
        let path = dir.join("manifest.json");
        std::fs::write(
            &path,
            serde_json::to_vec(&manifest).expect("manifest serialises"),
        )
        .expect("write manifest");
        path
    }

    fn open(dir: &Path) -> WindowCorpus {
        write_corpus(dir, TOOLCHAIN_ID);
        WindowCorpus::open_pinned(ARCH.to_owned(), sha256_hex(TOOLCHAIN_ID.as_bytes()), dir)
            .expect("corpus opens")
    }

    fn plan(entries: &[(&str, &str)]) -> Vec<(String, String)> {
        entries
            .iter()
            .map(|(s, a)| ((*s).to_owned(), (*a).to_owned()))
            .collect()
    }

    #[test]
    fn admit_manifest_accepts_the_exact_plan() {
        let dir = scratch("admit");
        let corpus = open(&dir);
        let path = write_manifest(
            &dir,
            vec![record(0, "alpha", &sha('7')), record(1, "beta", &sha('8'))],
        );
        assert_eq!(corpus.arch(), ARCH);
        corpus
            .admit_manifest(&path, &plan(&[("alpha", &sha('7')), ("beta", &sha('8'))]))
            .expect("exact plan admitted");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn admit_manifest_refuses_wrong_digest_and_wrong_order_naming_the_node() {
        let dir = scratch("refuse");
        let corpus = open(&dir);
        let path = write_manifest(
            &dir,
            vec![record(0, "alpha", &sha('7')), record(1, "beta", &sha('8'))],
        );
        let wrong_digest = corpus
            .admit_manifest(&path, &plan(&[("alpha", &sha('7')), ("beta", &sha('9'))]))
            .expect_err("a different object is refused");
        assert!(wrong_digest.contains("node 1 beta"), "{wrong_digest}");
        assert!(wrong_digest.contains("certified object"), "{wrong_digest}");
        let wrong_order = corpus
            .admit_manifest(&path, &plan(&[("beta", &sha('8')), ("alpha", &sha('7'))]))
            .expect_err("a reordered plan is refused");
        assert!(wrong_order.contains("node 0 beta"), "{wrong_order}");
        assert!(
            wrong_order.contains("manifest names alpha"),
            "{wrong_order}"
        );
        let truncated = corpus
            .admit_manifest(&path, &plan(&[("alpha", &sha('7'))]))
            .expect_err("a short plan is refused");
        assert!(truncated.contains("plan has 1 nodes"), "{truncated}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn admit_manifest_summarises_refusals_past_the_shown_limit() {
        let dir = scratch("many");
        let corpus = open(&dir);
        let records: Vec<_> = (0..12)
            .map(|i| record(i, &format!("k{i}"), &sha('7')))
            .collect();
        let path = write_manifest(&dir, records);
        let wrong: Vec<(String, String)> = (0..12).map(|i| (format!("k{i}"), sha('6'))).collect();
        let error = corpus
            .admit_manifest(&path, &wrong)
            .expect_err("every node is the wrong object");
        assert!(error.contains("(12 refusals)"), "{error}");
        assert!(error.contains("and 4 more"), "{error}");
        assert!(error.contains("node 7 k7"), "{error}");
        assert!(!error.contains("node 8 k8"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn write_named_manifest(dir: &Path, name: &str, nodes: Vec<WindowNodeRecord>) {
        let manifest = WindowManifest {
            schema: WINDOW_MANIFEST_SCHEMA.into(),
            version: WINDOW_MANIFEST_VERSION,
            shape: WHOLE16_SHAPE.into(),
            arch: ARCH.into(),
            toolchain_pin: sha256_hex(TOOLCHAIN_ID.as_bytes()),
            sequence_sha256: sequence_digest(&nodes),
            nodes,
            abi: Vec::new(),
        };
        std::fs::write(
            dir.join(name),
            serde_json::to_vec(&manifest).expect("manifest serialises"),
        )
        .expect("write manifest");
    }

    #[test]
    fn a_manifest_directory_admits_each_shape_by_its_own_sequence() {
        let dir = scratch("dir");
        let mut corpus = open(&dir);
        let manifests = dir.join("manifests");
        std::fs::create_dir_all(&manifests).unwrap();
        // Two shape families: a longer and a shorter node sequence.
        write_named_manifest(
            &manifests,
            "a.manifest.json",
            vec![record(0, "alpha", &sha('7')), record(1, "beta", &sha('8'))],
        );
        write_named_manifest(
            &manifests,
            "b.manifest.json",
            vec![
                record(0, "alpha", &sha('7')),
                record(1, "alpha", &sha('7')),
                record(2, "beta", &sha('8')),
            ],
        );
        let short = plan(&[("alpha", &sha('7')), ("beta", &sha('8'))]);
        let long = plan(&[
            ("alpha", &sha('7')),
            ("alpha", &sha('7')),
            ("beta", &sha('8')),
        ]);
        corpus
            .admit_manifest(&manifests, &short)
            .expect("short family");
        corpus
            .admit_manifest(&manifests, &long)
            .expect("long family");
        corpus
            .admit_manifest_witnessed(&manifests, &short, &[])
            .expect("short, witnessed");
        corpus
            .admit_manifest_witnessed(&manifests, &long, &[])
            .expect("long, witnessed");
        // An unknown sequence is one refusal naming the sequence, not a node dump.
        let unknown = plan(&[("alpha", &sha('7')), ("gamma", &sha('9'))]);
        let e = corpus.admit_manifest(&manifests, &unknown).unwrap_err();
        assert!(
            e.contains("no window manifest of the 2 loaded covers"),
            "{e}"
        );
        assert!(!e.contains("refusals"), "{e}");
        let e = corpus
            .admit_manifest_witnessed(&manifests, &unknown, &[])
            .unwrap_err();
        assert!(e.contains("covers this window's node sequence"), "{e}");
        // The same symbols over a different object are another sequence.
        let swapped = plan(&[("alpha", &sha('6')), ("beta", &sha('8'))]);
        assert!(corpus.admit_manifest(&manifests, &swapped).is_err());
        // An empty directory is refused as such.
        let empty = dir.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(corpus
            .admit_manifest(&empty, &short)
            .unwrap_err()
            .contains("holds no *.json manifest"));
    }

    #[test]
    fn the_mismatch_report_places_the_buffer_among_the_bound_allocations() {
        let kernargs = vec![
            (0x10_0000u64, 280usize, 4096usize),
            (0x10_1000, 280, 4096),
            (0x20_0000, 64, 4096),
        ];
        let f = SelfCheckFailure {
            first: "node 1 k".into(),
            kernarg: vec![KernargMismatch {
                node: 1,
                symbol: "k".into(),
                address: 0x10_1000,
                length: 280,
                allocation: 4096,
                offset: 0,
                expected: 0x7860_0000,
                actual: 0xbe31_c227,
                differing_bytes: 3,
                expected_head: vec![0, 0, 0x60, 0x78],
                actual_head: vec![0x27, 0xc2, 0x31, 0xbe],
            }],
        };
        // r0 ends 0x40 bytes below the buffer, r1 starts 0x100 above its
        // allocation, r2 overlaps it.
        let resources = [
            (0x10_0000u64, 0x1000 - 0x40),
            (0x10_2100, 256),
            (0x10_1800, 0x100),
        ];
        // Node 0's image holds a pointer 8 bytes below the buffer.
        let mut image0 = vec![0u8; 16];
        image0[8..16].copy_from_slice(&(0x10_1000u64 - 8).to_le_bytes());
        let images: Vec<&[u8]> = vec![&image0, &[], &[]];
        let names = vec![
            "".to_owned(),
            "".to_owned(),
            "fp8_verify_x8_scratch".to_owned(),
        ];
        let report = mismatch_report(&f, &resources, &names, &kernargs, &images);
        let all = report.join("\n");
        assert!(report[0].contains("1 of 3 kernarg buffers differ"), "{all}");
        assert!(all.contains("buffer 0x101000+280 (alloc 4096)"), "{all}");
        assert!(all.contains("expected 00006078 actual 27c231be"), "{all}");
        assert!(
            all.contains("overlaps program resources: r2(fp8_verify_x8_scratch)[0x101800+256]"),
            "{all}"
        );
        assert!(all.contains("r0 ends 64 B below"), "{all}");
        assert!(all.contains("r1 starts 256 B above"), "{all}");
        assert!(all.contains("node 0 +8=0x100ff8"), "{all}");
        assert!(all.contains("kernarg buffers within 64 KiB: [0]"), "{all}");
    }

    #[test]
    fn a_single_manifest_still_reports_node_by_node() {
        let dir = scratch("single-detail");
        let corpus = open(&dir);
        let path = write_manifest(
            &dir,
            vec![record(0, "alpha", &sha('7')), record(1, "beta", &sha('8'))],
        );
        let e = corpus
            .admit_manifest(&path, &plan(&[("alpha", &sha('7')), ("gamma", &sha('8'))]))
            .unwrap_err();
        assert!(e.contains("node 1 gamma"), "{e}");
    }

    #[test]
    fn open_refuses_a_corpus_from_another_toolchain() {
        let dir = scratch("pin");
        write_corpus(&dir, "AMD clang version other");
        let error =
            WindowCorpus::open_pinned(ARCH.to_owned(), sha256_hex(TOOLCHAIN_ID.as_bytes()), &dir)
                .err()
                .expect("a foreign toolchain pin is refused");
        assert!(error.contains("is not the corpus pin"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
