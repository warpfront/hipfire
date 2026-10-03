//! railgun M1 shadow mode (railgun design §7 M1, §5 G7).
//!
//! With `HIPFIRE_RAILGUN_SHADOW=<corpus dir>` every launch Redline records is
//! also observed for railgun: the SHA-256 of the object it ran is looked up
//! in the JIT receipt corpus (§2.5), its A1 pointer arguments are bound to
//! the live allocations, and the launch site's declared words are kept. When
//! Redline prepares its single-IB gfx12 tape, railgun authors its own
//! program from those facts, derives edges and boundaries from its cache
//! table, lowers it to PM4 command dwords and kernarg images (same kernarg
//! pool, same loaded kernels; nothing is submitted) and diffs the result
//! against Redline's prepared tape. The route that executes is unchanged.
//!
//! `HIPFIRE_RAILGUN_SHADOW_OUT=<dir>` receives one JSON report per prepare;
//! `HIPFIRE_RAILGUN_INVENTORY=<file>` is the MR recording-predicate
//! inventory (`railgun-cert recording-inventory json`), whose decisions for
//! the matched route go into the program certificate.
//!
//! M2: `HIPFIRE_RAILGUN_CACHE_TABLE=<cache-table.<arch>.json>` attaches the
//! G4 silicon receipts (`railgun/examples/g4_probe`) to the cache table rows
//! they back. `HIPFIRE_RAILGUN_BACKEND=railgun` makes the prepared tape
//! *execute* railgun's lowering (its command stream, over Redline's kernarg
//! buffers once every railgun kernarg image equals Redline's byte for byte);
//! any refusal fails the prepare closed (the forward falls to HIP, never to
//! the Redline planner).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use hip_bridge::HipRuntime;
use railgun::kernel::{self as rk, DeclaredWord, LaunchRecord, Lowering, NodeFacts};
use railgun::plan::{self as rp, CacheTable, Pm4Op, RowPolicy, Rung};
use railgun::shadow::{self as rs, RailgunLowering, RedlineDecision, RedlineDispatch, RedlineTape, SegmentIb, SlotBinding};
use railgun::Binding;
use railgun_corpus::{sha256_hex, ArtifactKey, JitCorpus, Lookup, ProgramNode};
use serde::Serialize;

use super::{
    Gfx11EntryAcquirePolicy, Gfx12DispatchPacing, KernargBuffer, KernargPool, Kernel, LaunchGeometry, Pm4Commands,
    RecordedHipLaunch, ReplayGridBinding, ReplayKernargBinding,
};
use crate::compiler::KernelCompiler;

/// The MR inventory names programs by route (`railgun-cert` `Program`), the
/// corpus by admitted default (`railgun-jitcorpus` `programs.json`).
const INVENTORY_PROGRAM: &[(&str, &str)] = &[
    ("qwen35-dense-h2-gfx1201", "h2_gfx1201"),
    ("qwen36-a3b-mq4r-gfx1201", "mq4r_gfx1201"),
    ("qwen36-a3b-mq4r-gfx1100", "mq4r_gfx1100"),
    ("qwen36-a3b-mq4r-gfx1151", "mq4r_gfx1151"),
    ("deepseek4-mq2r-gfx1151", "ds4_gfx1151"),
];

struct ShadowConfig {
    corpus: PathBuf,
    out: Option<PathBuf>,
    inventory: Option<PathBuf>,
    cache_table: Option<PathBuf>,
    backend_railgun: bool,
}

/// What check mode needs from the prepared railgun program.
#[derive(Clone, Debug, Default)]
pub(crate) struct ProgramSurfaces {
    /// Allocations some node writes (A1 effects; allocation-wide in M1).
    pub written: Vec<Binding>,
    /// Nodes not lowered to PM4 (HipDirect/Graph); `sample` refuses them.
    pub non_pm4_nodes: usize,
    /// The railgun lowering is what executes (`HIPFIRE_RAILGUN_BACKEND=railgun`).
    pub executing: bool,
}

/// Minimum probe trials per rung for a G4 receipt (design §5 G4 asks 10⁴;
/// the M2 probe runs 2²⁰).
const G4_MIN_TRIALS: u64 = 1 << 20;

/// Recorder-side state of the shadow: one [`LaunchRecord`] per launch
/// Redline recorded, in order.
pub(super) struct RailgunShadow {
    config: ShadowConfig,
    corpus: Option<Result<Arc<JitCorpus>, String>>,
    artifact_sha: HashMap<PathBuf, Result<String, String>>,
    facts: HashMap<(String, String), Result<Arc<NodeFacts>, String>>,
    launches: Vec<LaunchRecord>,
    /// SHA-256 of each launch's object (for route matching).
    artifacts: Vec<Option<String>>,
    arch: Option<String>,
    toolchain_pin: Option<String>,
    reports: usize,
    surfaces: Option<ProgramSurfaces>,
}

/// Redline's prepared single-IB tape, as the lowering loop left it.
pub(super) struct RedlinePrepared<'a> {
    pub device_name: &'a str,
    pub launches: &'a [RecordedHipLaunch],
    pub ib: &'a [u32],
    pub kernels: &'a [Kernel],
    /// Loader kernarg image and buffer address per dispatch.
    pub kernargs: Vec<(Vec<u8>, u64)>,
    pub decisions: Vec<RedlineDecision>,
    pub word_patches: &'a [(usize, ReplayKernargBinding)],
    pub grid_patches: &'a [(usize, ReplayGridBinding, [u32; 3], [u32; 3])],
    /// Redline's tape resources: id → `(allocation_base, allocation_bytes)`.
    pub resources: BTreeMap<u32, (u64, u64)>,
}

/// The transport railgun lowers through: Redline's, reused unchanged
/// (design §4.2).
pub(super) struct Transport<'a> {
    pub pool: &'a KernargPool,
    pub new_commands: &'a dyn Fn() -> Pm4Commands,
    pub gcr_trim: bool,
    pub entry_policy: Gfx11EntryAcquirePolicy,
}

impl RailgunShadow {
    pub(super) fn from_config() -> Option<Self> {
        let var = |name: &str| hipfire_config::developer_var(name).ok().filter(|v| !v.is_empty());
        let corpus = var("HIPFIRE_RAILGUN_SHADOW").filter(|v| v != "0")?;
        Some(Self {
            config: ShadowConfig {
                corpus: corpus.into(),
                out: var("HIPFIRE_RAILGUN_SHADOW_OUT").map(PathBuf::from),
                inventory: var("HIPFIRE_RAILGUN_INVENTORY").map(PathBuf::from),
                cache_table: var("HIPFIRE_RAILGUN_CACHE_TABLE").map(PathBuf::from),
                backend_railgun: var("HIPFIRE_RAILGUN_BACKEND").as_deref() == Some("railgun"),
            },
            corpus: None,
            artifact_sha: HashMap::new(),
            facts: HashMap::new(),
            launches: Vec::new(),
            artifacts: Vec::new(),
            arch: None,
            toolchain_pin: None,
            reports: 0,
            surfaces: None,
        })
    }

    /// `HIPFIRE_RAILGUN_BACKEND=railgun`: the prepared tape executes railgun's lowering.
    pub(super) fn backend_railgun(&self) -> bool {
        self.config.backend_railgun
    }

    /// The last prepared program's written allocations (check mode `sample`).
    pub(crate) fn surfaces(&self) -> Option<&ProgramSurfaces> {
        self.surfaces.as_ref()
    }

    /// A new tape starts.
    pub(super) fn reset(&mut self) {
        self.launches.clear();
        self.artifacts.clear();
    }

    /// Observe one launch Redline just recorded.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe(
        &mut self,
        hip: &HipRuntime,
        compiler: Option<&KernelCompiler>,
        kernel: &str,
        artifact: Option<&Path>,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        kernarg: &[u8],
        declared: &[DeclaredWord],
    ) {
        if let Some(compiler) = compiler {
            self.arch.get_or_insert_with(|| compiler.arch().to_owned());
            self.toolchain_pin.get_or_insert_with(|| sha256_hex(compiler.toolchain_id().as_bytes()));
        }
        let sha = artifact.map(|path| {
            self.artifact_sha
                .entry(path.to_path_buf())
                .or_insert_with(|| {
                    std::fs::read(path).map(|bytes| sha256_hex(&bytes)).map_err(|e| format!("{}: {e}", path.display()))
                })
                .clone()
        });
        let facts = match &sha {
            None => Err("the launch has no owning code object".to_owned()),
            Some(Err(e)) => Err(e.clone()),
            Some(Ok(sha)) => self.facts_for(sha, kernel),
        };
        let allocations = match &facts {
            Ok(f) => f
                .facts
                .pointer_offsets()
                .filter_map(|offset| {
                    let bytes = kernarg.get(offset as usize..offset as usize + 8)?;
                    let value = u64::from_le_bytes(bytes.try_into().ok()?);
                    (value != 0).then(|| (offset, probe(hip, value)))
                })
                .collect(),
            Err(_) => Vec::new(),
        };
        self.artifacts.push(sha.and_then(Result::ok));
        self.launches.push(LaunchRecord {
            symbol: kernel.to_owned(),
            grid,
            block,
            dynamic_lds: shared_mem,
            kernarg: kernarg.to_vec(),
            declared: declared.to_vec(),
            facts,
            allocations,
        });
    }

    fn open_corpus(&mut self) -> Result<Arc<JitCorpus>, String> {
        self.corpus
            .get_or_insert_with(|| {
                JitCorpus::open(&self.config.corpus).map(Arc::new).map_err(|e| format!("railgun JIT corpus rejected: {e}"))
            })
            .clone()
    }

    fn facts_for(&mut self, sha: &str, symbol: &str) -> Result<Arc<NodeFacts>, String> {
        let key = (sha.to_owned(), symbol.to_owned());
        if let Some(hit) = self.facts.get(&key) {
            return hit.clone();
        }
        let result = (|| {
            let corpus = self.open_corpus()?;
            let arch = self.arch.as_deref().ok_or("no compiler arch was observed")?;
            let pin = self.toolchain_pin.as_deref().ok_or("no compiler toolchain was observed")?;
            match corpus.lookup(&ArtifactKey { arch, toolchain_pin: pin, artifact_sha256: sha }) {
                Ok(Lookup::Hit(hit)) => NodeFacts::from_hit(&hit.record, &hit.receipt, symbol).map(Arc::new),
                Ok(Lookup::Missing) => Err(match corpus.rejection(arch, sha) {
                    Some(r) => format!("railgun-cert rejected {}: {}", &sha[..16], r.reason),
                    None if pin != corpus.toolchain_pin() => {
                        format!("toolchain pin {} is not the corpus pin {}", &pin[..16], &corpus.toolchain_pin()[..16])
                    }
                    None => format!("no record for {arch} object {}", &sha[..16]),
                }),
                Err(e) => Err(e.to_string()),
            }
        })();
        self.facts.insert(key, result.clone());
        result
    }

    /// Author, plan, lower and diff; report. In shadow mode this never fails
    /// Redline's prepare. With `HIPFIRE_RAILGUN_BACKEND=railgun` it returns
    /// the executable lowering (`Some(Ok)`) or the refusal (`Some(Err)`).
    pub(super) fn compare(
        &mut self,
        redline: RedlinePrepared<'_>,
        transport: Transport<'_>,
    ) -> Option<Result<Pm4Commands, String>> {
        let started = Instant::now();
        self.reports += 1;
        self.surfaces = None;
        let mut executable = None;
        let report = self.compare_inner(&redline, &transport, started, &mut executable);
        if self.config.backend_railgun && executable.is_none() {
            executable = Some(Err(match &report {
                Ok(_) => "railgun backend produced no lowering".to_owned(),
                Err(e) => e.clone(),
            }));
        }
        let (summary, value) = match report {
            Ok(report) => (report.summary(), serde_json::to_value(&report)),
            Err(error) => (
                format!("[railgun] shadow FAILED: {error}"),
                serde_json::to_value(serde_json::json!({ "schema": "railgun-m1-shadow", "version": 1, "error": error })),
            ),
        };
        let path = self.config.out.as_ref().map(|dir| dir.join(format!("railgun-shadow-{}-{}.json", std::process::id(), self.reports)));
        let written = match (&path, value) {
            (Some(path), Ok(value)) => std::fs::create_dir_all(path.parent().expect("file in a directory"))
                .and_then(|_| std::fs::write(path, serde_json::to_vec_pretty(&value).expect("JSON value serialises")))
                .map(|_| format!(" report={}", path.display()))
                .unwrap_or_else(|e| format!(" report write failed: {e}")),
            (Some(_), Err(e)) => format!(" report serialisation failed: {e}"),
            (None, _) => String::new(),
        };
        eprintln!("{summary}{written}");
        match &executable {
            Some(Ok(commands)) => {
                if let Some(s) = self.surfaces.as_mut() {
                    s.executing = true;
                }
                eprintln!("[railgun] backend=railgun: executing railgun's lowering ({} dwords)", commands.len_dwords());
            }
            Some(Err(reason)) => eprintln!("[railgun] backend=railgun refused: {reason}"),
            None => {}
        }
        executable
    }

    fn compare_inner(
        &mut self,
        redline: &RedlinePrepared<'_>,
        transport: &Transport<'_>,
        started: Instant,
        executable: &mut Option<Result<Pm4Commands, String>>,
    ) -> Result<Report, String> {
        let n = redline.launches.len();
        if self.launches.len() != n {
            return Err(format!("railgun observed {} launches, Redline prepared {n}", self.launches.len()));
        }
        for (k, (ours, theirs)) in self.launches.iter().zip(redline.launches).enumerate() {
            if ours.symbol != theirs.kernel || ours.kernarg != theirs.kernarg {
                return Err(format!("launch {k}: railgun observed {} but Redline recorded {}", ours.symbol, theirs.kernel));
            }
        }
        let arch = self.arch.clone().unwrap_or_else(|| redline.device_name.to_owned());
        let mut table = CacheTable::for_arch(&arch).ok_or_else(|| format!("railgun M1 has no cache table for {arch}"))?;
        let receipts = match &self.config.cache_table {
            Some(path) => {
                let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
                let probe: rp::G4Table = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
                let source = format!("{} sha256={}", path.display(), sha256_hex(&bytes));
                let attached = table.attach_receipts(&probe, &source, G4_MIN_TRIALS)?;
                eprintln!("[railgun] G4 receipts: {attached}/{} {arch} rows backed by {source}", table.rows.len());
                attached
            }
            None => 0,
        };
        let _ = receipts;
        let program = rk::author(&arch, &self.launches);
        let plan = rp::plan(&program, &table, RowPolicy::TierD);
        let strict = rp::plan(&program, &table, RowPolicy::UnreceiptedSystem);
        let arch_lowering = arch_lowering(&arch);
        let (lowering, _buffers) = lower(&program, &plan, arch_lowering, redline.kernels, transport)?;
        let tape = redline_tape(redline);
        let diff = rs::diff(&program, &plan, &lowering, &tape);
        let mut written: Vec<Binding> = Vec::new();
        for node in &program.nodes {
            for effect in node.effects.iter().filter(|e| e.writes()) {
                let binding = program.resources[effect.resource.0 as usize];
                if !written.contains(&binding) {
                    written.push(binding);
                }
            }
        }
        self.surfaces = Some(ProgramSurfaces {
            written,
            non_pm4_nodes: program.nodes.iter().filter(|n| !n.lowering.is_pm4()).count(),
            executing: false,
        });
        if self.config.backend_railgun {
            *executable = Some(lower_executable(&program, &plan, arch_lowering, &lowering, redline, transport));
        }

        let corpus = self.open_corpus().ok();
        let route = corpus.as_deref().and_then(|c| self.match_route(c, &arch));
        let program_check = match (&corpus, &route, &self.toolchain_pin) {
            (Some(c), Some(id), Some(pin)) => {
                let pairs: BTreeSet<(String, String)> = self
                    .launches
                    .iter()
                    .zip(&self.artifacts)
                    .filter_map(|(l, sha)| sha.clone().map(|sha| (l.symbol.clone(), sha)))
                    .collect();
                let nodes: Vec<ProgramNode> = pairs.iter().map(|(s, a)| ProgramNode { symbol: s, artifact_sha256: a }).collect();
                match c.check_program(id, &arch, pin, &nodes) {
                    Ok(()) => "ok".to_owned(),
                    Err(e) => e.to_string(),
                }
            }
            _ => "no admitted program matches the tape".to_owned(),
        };
        let inventory = route.as_deref().map(|id| self.inventory(id));
        let facts = facts_summary(&program);
        let obligations = obligations(&program, &plan, &table);
        let resources_overlapping = overlapping(&program.resources);
        let redline_effects = effect_sources(&redline.decisions);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        let tape_hash = super::replay_sequence_hash(redline.launches);
        Ok(Report {
            schema: "railgun-m1-shadow",
            version: 1,
            arch,
            device: redline.device_name.to_owned(),
            tape: TapeSummary {
                launches: n,
                unique_kernels: redline.launches.iter().map(|l| l.kernel.as_str()).collect::<BTreeSet<_>>().len(),
                sequence_hash: format!("{tape_hash:016x}"),
            },
            corpus: corpus.as_deref().map(|c| CorpusSummary {
                root: c.root().display().to_string(),
                source_commit: c.index().source_commit.clone(),
                build_host: c.index().build_host.key.clone(),
                toolchain: c.index().toolchain.id.clone(),
            }),
            route,
            program_check,
            inventory,
            facts,
            program: ProgramSummary {
                nodes: program.nodes.len(),
                resources: program.resources.len(),
                words: program.words.clone(),
                resources_overlapping,
            },
            certificate: Certificate { tier: "D", obligations, rows_used: plan.rows_used.clone(), cache_table: table, lowering: arch_lowering },
            plan: PlanSummary { tier_d: plan.counts(), unreceipted_system: strict.counts() },
            redline_effects,
            diff,
            elapsed_ms,
        })
    }

    fn match_route(&self, corpus: &JitCorpus, arch: &str) -> Option<String> {
        let tape: BTreeSet<(&str, &str)> = self
            .launches
            .iter()
            .zip(&self.artifacts)
            .map(|(l, sha)| (l.symbol.as_str(), sha.as_deref().unwrap_or("")))
            .collect();
        corpus
            .index()
            .programs
            .iter()
            .filter(|p| p.arch == arch)
            .find(|p| p.nodes.iter().map(|n| (n.symbol.as_str(), n.artifact_sha256.as_str())).collect::<BTreeSet<_>>() == tape)
            .map(|p| p.id.clone())
    }

    fn inventory(&self, route: &str) -> InventoryEntry {
        let Some(path) = &self.config.inventory else {
            return InventoryEntry { program: None, error: Some("HIPFIRE_RAILGUN_INVENTORY is not set".into()), ..InventoryEntry::default() };
        };
        let Some((_, program)) = INVENTORY_PROGRAM.iter().find(|(id, _)| *id == route) else {
            return InventoryEntry { program: None, error: Some(format!("route {route} has no inventory program")), ..InventoryEntry::default() };
        };
        let parsed = std::fs::read(path)
            .map_err(|e| format!("{}: {e}", path.display()))
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|e| format!("{}: {e}", path.display())));
        match parsed {
            Ok(value) => {
                let entry = &value["programs"][*program];
                InventoryEntry {
                    program: Some((*program).to_owned()),
                    recording_dependent_decisions: entry["recording_dependent_decisions"].clone(),
                    refused_as_default: entry["refused_as_default"].as_bool(),
                    error: entry.is_null().then(|| format!("inventory has no program {program}")),
                }
            }
            Err(error) => InventoryEntry { program: Some((*program).to_owned()), error: Some(error), ..InventoryEntry::default() },
        }
    }
}

fn probe(hip: &HipRuntime, value: u64) -> Result<Binding, String> {
    hip.mem_get_address_range(value as usize as *mut std::ffi::c_void)
        .map(|(base, bytes)| Binding { base: base as usize as u64, bytes: bytes as u64 })
        .map_err(|e| e.to_string())
}

/// railgun's per-arch lowering parameters for `arch`. On gfx1201 an explicit
/// `HIPFIRE_GFX1201_PM4_PACING` (`off` / `nop:N`, the documented knob of the
/// gfx1201 tape pacing) replaces the arch default; unset or `auto` keeps it.
fn arch_lowering(arch: &str) -> rp::ArchLowering {
    let mut lowering = rp::ArchLowering::for_arch(arch);
    if arch == "gfx1201" {
        if let Some(pacing) = super::gfx1201_pm4_pacing_override() {
            lowering.post_dispatch_nop = match pacing {
                Gfx12DispatchPacing::None => 0,
                Gfx12DispatchPacing::PostDispatchNop(n) => n,
            };
        }
    }
    lowering
}

/// A fresh command buffer from Redline's builder, paced per `lowering`.
fn paced_commands(transport: &Transport<'_>, lowering: rp::ArchLowering) -> Pm4Commands {
    let mut commands = (transport.new_commands)();
    if let Pm4Commands::Gfx12(gfx12) = &mut commands {
        gfx12.set_dispatch_pacing(match lowering.post_dispatch_nop {
            0 => Gfx12DispatchPacing::None,
            n => Gfx12DispatchPacing::PostDispatchNop(n),
        });
    }
    commands
}

/// Lower every segment through Redline's command builder with the arch's
/// lowering parameters; allocate and fill railgun's kernarg buffers. The
/// buffers are returned so they outlive the diff; nothing is submitted.
fn lower(
    program: &rk::KernelProgram,
    plan: &rp::Pm4Plan,
    arch_lowering: rp::ArchLowering,
    kernels: &[Kernel],
    transport: &Transport<'_>,
) -> Result<(RailgunLowering, Vec<KernargBuffer>), String> {
    let words = program.authored_words();
    let mut buffers = Vec::new();
    let mut kernargs = vec![None; program.nodes.len()];
    let mut segments = Vec::with_capacity(plan.segments.len());
    for segment in &plan.segments {
        let mut commands = paced_commands(transport, arch_lowering);
        for op in plan.segment_ops(segment) {
            match op {
                Pm4Op::Entry => {
                    commands.emit_entry_sentinel_reset()?;
                    commands.acquire_entry(transport.gcr_trim, transport.entry_policy);
                }
                Pm4Op::Boundary(v) => {
                    if v.wait_idle {
                        commands.wait_compute_idle()?;
                    }
                    match v.acquire {
                        Rung::None => {}
                        Rung::InterNode => commands.acquire_inter_node(transport.gcr_trim, false),
                        Rung::System => commands.gfx12_system_acquire()?,
                    }
                }
                Pm4Op::Dispatch(i) => {
                    let node = &program.nodes[i];
                    let kernel = kernels.get(i).ok_or_else(|| format!("no loaded kernel for node {i}"))?;
                    let metadata = kernel.metadata();
                    let image = program.kernarg_image(i, metadata.kernarg_segment_size as usize, &words)?;
                    let mut buffer = transport.pool.allocate_for(metadata).map_err(|e| format!("{}: kernarg: {e}", node.symbol))?;
                    buffer.write_exact(&image).map_err(|e| format!("{}: kernarg: {e}", node.symbol))?;
                    let mut workgroup = [0u16; 3];
                    for axis in 0..3 {
                        workgroup[axis] = u16::try_from(node.block[axis]).map_err(|_| format!("{}: block exceeds u16", node.symbol))?;
                    }
                    let geometry = LaunchGeometry::from_hip_workgroups(node.grid, workgroup).map_err(|e| format!("{}: {e}", node.symbol))?;
                    commands.dispatch(kernel, geometry, node.dynamic_lds, buffer.address()).map_err(|e| format!("{}: {e}", node.symbol))?;
                    kernargs[i] = Some((image, buffer.address() as usize as u64));
                    buffers.push(buffer);
                }
                Pm4Op::Exit => commands.trailing_release()?,
            }
        }
        let ib = commands.gfx12_dwords().ok_or("railgun M1 lowers gfx12 command buffers only")?.to_vec();
        segments.push(SegmentIb { first: segment.first, end: segment.end, ib });
    }
    Ok((RailgunLowering { segments, kernargs, post_dispatch_nop: arch_lowering.post_dispatch_nop }, buffers))
}

/// The executable railgun lowering (`HIPFIRE_RAILGUN_BACKEND=railgun`): one
/// segment spanning the whole tape, emitted through Redline's command
/// builder with dispatches pointing at Redline's kernarg buffers. Admitted
/// only when railgun's own kernarg image of every node equals the bytes in
/// that buffer, so what runs is railgun's program; Redline's binding patches
/// (position, GDN frame words) then apply unchanged.
fn lower_executable(
    program: &rk::KernelProgram,
    plan: &rp::Pm4Plan,
    arch_lowering: rp::ArchLowering,
    lowering: &RailgunLowering,
    redline: &RedlinePrepared<'_>,
    transport: &Transport<'_>,
) -> Result<Pm4Commands, String> {
    let n = program.nodes.len();
    match plan.segments.as_slice() {
        [only] if only.first == 0 && only.end == n => {}
        segments => {
            return Err(format!(
                "M2 executes single-segment programs only ({} segments, {} non-PM4 nodes)",
                segments.len(),
                program.nodes.iter().filter(|x| !x.lowering.is_pm4()).count()
            ))
        }
    }
    for (i, (ours, theirs)) in lowering.kernargs.iter().zip(&redline.kernargs).enumerate() {
        let Some((image, _)) = ours else { return Err(format!("node {i} has no railgun kernarg image")) };
        if *image != theirs.0 {
            let at = image.iter().zip(&theirs.0).position(|(a, b)| a != b).unwrap_or(image.len().min(theirs.0.len()));
            return Err(format!("node {i} {}: railgun kernarg image differs from Redline's at byte {at}", program.nodes[i].symbol));
        }
    }
    // Check-mode negative control (developer only): drop every mid-segment
    // boundary so dependent dispatches race; the HIP twin must then differ.
    let drop_boundaries = hipfire_config::developer_var("HIPFIRE_RAILGUN_NEGATIVE_CONTROL").as_deref() == Ok("drop_boundaries");
    if drop_boundaries {
        eprintln!("[railgun] NEGATIVE CONTROL: executable lowering without mid-segment boundaries");
    }
    let mut commands = paced_commands(transport, arch_lowering);
    for op in plan.segment_ops(&plan.segments[0]) {
        match op {
            Pm4Op::Entry => {
                commands.emit_entry_sentinel_reset()?;
                commands.acquire_entry(transport.gcr_trim, transport.entry_policy);
            }
            Pm4Op::Boundary(_) if drop_boundaries => {}
            Pm4Op::Boundary(v) => {
                if v.wait_idle {
                    commands.wait_compute_idle()?;
                }
                match v.acquire {
                    Rung::None => {}
                    Rung::InterNode => commands.acquire_inter_node(transport.gcr_trim, false),
                    Rung::System => commands.gfx12_system_acquire()?,
                }
            }
            Pm4Op::Dispatch(i) => {
                let node = &program.nodes[i];
                let kernel = redline.kernels.get(i).ok_or_else(|| format!("no loaded kernel for node {i}"))?;
                let mut workgroup = [0u16; 3];
                for axis in 0..3 {
                    workgroup[axis] = u16::try_from(node.block[axis]).map_err(|_| format!("{}: block exceeds u16", node.symbol))?;
                }
                let geometry = LaunchGeometry::from_hip_workgroups(node.grid, workgroup).map_err(|e| format!("{}: {e}", node.symbol))?;
                let address = redline.kernargs[i].1 as usize as *mut std::ffi::c_void;
                commands.dispatch(kernel, geometry, node.dynamic_lds, address).map_err(|e| format!("{}: {e}", node.symbol))?;
            }
            Pm4Op::Exit => commands.trailing_release()?,
        }
    }
    let ours = commands.gfx12_dwords().ok_or("railgun backend lowers gfx12 command buffers only")?;
    let shadow = &lowering.segments[0].ib;
    if ours.len() != shadow.len() && !drop_boundaries {
        return Err(format!("executable lowering has {} dwords, the shadow lowering {}", ours.len(), shadow.len()));
    }
    Ok(commands)
}

fn redline_tape(redline: &RedlinePrepared<'_>) -> RedlineTape {
    let dispatches = redline
        .launches
        .iter()
        .zip(&redline.kernargs)
        .zip(&redline.decisions)
        .map(|((launch, (kernarg, address)), decision)| RedlineDispatch {
            symbol: launch.kernel.clone(),
            kernarg: kernarg.clone(),
            kernarg_address: *address,
            bindings: launch
                .binding_layout
                .iter()
                .flat_map(|layout| &layout.slots)
                .filter_map(|slot| {
                    redline.resources.get(&slot.resource.index()).map(|&(base, bytes)| SlotBinding {
                        offset: slot.offset as u32,
                        base,
                        bytes,
                        interior: slot.interior_offset,
                    })
                })
                .collect(),
            decision: decision.clone(),
        })
        .collect();
    let word_patches = redline
        .word_patches
        .iter()
        .map(|(dispatch, binding)| match binding {
            ReplayKernargBinding::GdnFrameU32 { offset, frames } => {
                rs::WordPatch { dispatch: *dispatch, offset: *offset as u32, kind: format!("gdn_frame(frames={frames})") }
            }
            ReplayKernargBinding::PositionPlusU32 { offset, addend } => {
                rs::WordPatch { dispatch: *dispatch, offset: *offset as u32, kind: format!("position+{addend}") }
            }
            // Bindings beta added after railgun M1: no railgun word role
            // derives them yet, so the shadow reports them as Redline-only.
            ReplayKernargBinding::PositionDivU32 { offset, addend, divisor } => rs::WordPatch {
                dispatch: *dispatch,
                offset: *offset as u32,
                kind: format!("(position+{addend})/{divisor}"),
            },
            ReplayKernargBinding::PositionModU32 { offset, addend, modulus } => rs::WordPatch {
                dispatch: *dispatch,
                offset: *offset as u32,
                kind: format!("(position+{addend})%{modulus}"),
            },
            ReplayKernargBinding::PositionMulU32 { offset, factor } => {
                rs::WordPatch { dispatch: *dispatch, offset: *offset as u32, kind: format!("position*{factor}") }
            }
        })
        .collect();
    let grid_patches = redline
        .grid_patches
        .iter()
        .map(|(dispatch, binding, _, _)| {
            let ReplayGridBinding::PositionCeilDiv { axis, addend, divisor } = *binding;
            rs::GridPatch { dispatch: *dispatch, axis, addend, divisor }
        })
        .collect();
    RedlineTape { ib: redline.ib.to_vec(), dispatches, word_patches, grid_patches }
}

fn facts_summary(program: &rk::KernelProgram) -> FactsSummary {
    let mut per_kernel: BTreeMap<(String, String), KernelFactsRow> = BTreeMap::new();
    let mut fully_certified = 0;
    let mut not_pm4 = Vec::new();
    for (k, node) in program.nodes.iter().enumerate() {
        let certified = node.kref.is_some() && node.lowering.is_pm4();
        fully_certified += usize::from(certified);
        let (lowering, reason) = match &node.lowering {
            Lowering::Pm4 => ("pm4", None),
            Lowering::Graph { reason } => ("graph", Some(reason.clone())),
            Lowering::HipDirect { reason } => ("hip_direct", Some(reason.clone())),
        };
        if let Some(reason) = &reason {
            not_pm4.push(format!("{k} {}: {reason}", node.symbol));
        }
        let row = per_kernel.entry((node.symbol.clone(), lowering.to_owned())).or_insert_with(|| KernelFactsRow {
            symbol: node.symbol.clone(),
            module: node.kref.as_ref().map(|r| r.module.clone()),
            artifact_sha256: node.kref.as_ref().map(|r| r.artifact_sha256.clone()),
            kd_sha256: node.kref.as_ref().map(|r| r.kd_sha256.clone()),
            tier: node.kref.as_ref().map(|r| r.tier.clone()),
            read_cache: node.read_cache,
            lowering: lowering.to_owned(),
            reason,
            launches: 0,
            pointer_args: node.args.iter().filter(|(_, a)| matches!(a, rk::Arg::Resource { .. })).count(),
            effects: node.effects.len(),
        });
        row.launches += 1;
    }
    FactsSummary { launches: program.nodes.len(), fully_certified, per_kernel: per_kernel.into_values().collect(), not_pm4 }
}

fn obligations(program: &rk::KernelProgram, plan: &rp::Pm4Plan, table: &CacheTable) -> BTreeSet<String> {
    let mut out: BTreeSet<String> = program.nodes.iter().filter_map(|n| n.kref.as_ref()).flat_map(|r| r.obligations.iter().cloned()).collect();
    if program.nodes.iter().any(|n| !n.effects.is_empty()) {
        out.insert("InBoundsNoAlias".to_owned());
    }
    for (hazard, _) in &plan.rows_used {
        let row = table.row(*hazard);
        if row.g4_receipt.is_none() {
            out.insert(format!("G4 {} {hazard:?} row (wait={}, acquire={:?})", table.arch, row.visibility.wait_idle, row.visibility.acquire));
        }
    }
    if !program.words.is_empty() {
        out.insert("GdnFrame word host-patched at submit (I6 device counter pending)".to_owned());
    }
    out
}

fn overlapping(resources: &[Binding]) -> Vec<String> {
    let mut sorted: Vec<&Binding> = resources.iter().collect();
    sorted.sort_by_key(|b| b.base);
    sorted
        .windows(2)
        .filter(|w| w[0].base + w[0].bytes > w[1].base)
        .map(|w| format!("{:#x}+{} overlaps {:#x}+{}", w[0].base, w[0].bytes, w[1].base, w[1].bytes))
        .collect()
}

fn effect_sources(decisions: &[RedlineDecision]) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for d in decisions {
        *out.entry(d.effects.clone()).or_insert(0) += 1;
    }
    out
}

#[derive(Serialize)]
struct TapeSummary {
    launches: usize,
    unique_kernels: usize,
    sequence_hash: String,
}

#[derive(Serialize)]
struct CorpusSummary {
    root: String,
    source_commit: String,
    build_host: String,
    toolchain: String,
}

#[derive(Default, Serialize)]
struct InventoryEntry {
    program: Option<String>,
    recording_dependent_decisions: serde_json::Value,
    refused_as_default: Option<bool>,
    error: Option<String>,
}

#[derive(Serialize)]
struct KernelFactsRow {
    symbol: String,
    module: Option<String>,
    artifact_sha256: Option<String>,
    kd_sha256: Option<String>,
    tier: Option<String>,
    read_cache: rk::ReadCache,
    lowering: String,
    reason: Option<String>,
    launches: usize,
    pointer_args: usize,
    effects: usize,
}

#[derive(Serialize)]
struct FactsSummary {
    launches: usize,
    /// Record hit, A1 Proven, G8 pass, every pointer bound, every word typed.
    fully_certified: usize,
    per_kernel: Vec<KernelFactsRow>,
    not_pm4: Vec<String>,
}

#[derive(Serialize)]
struct ProgramSummary {
    nodes: usize,
    resources: usize,
    words: Vec<rk::ProgramWord>,
    resources_overlapping: Vec<String>,
}

#[derive(Serialize)]
struct Certificate {
    tier: &'static str,
    obligations: BTreeSet<String>,
    rows_used: BTreeMap<rp::Hazard, usize>,
    cache_table: CacheTable,
    /// Per-arch lowering parameters (post-dispatch NOP pacing).
    lowering: rp::ArchLowering,
}

#[derive(Serialize)]
struct PlanSummary {
    tier_d: rp::PlanCounts,
    /// §2.2 read strictly: every unreceipted row at `WaitIdle + System`.
    unreceipted_system: rp::PlanCounts,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    version: u32,
    arch: String,
    device: String,
    tape: TapeSummary,
    corpus: Option<CorpusSummary>,
    route: Option<String>,
    program_check: String,
    inventory: Option<InventoryEntry>,
    facts: FactsSummary,
    program: ProgramSummary,
    certificate: Certificate,
    plan: PlanSummary,
    redline_effects: BTreeMap<String, usize>,
    diff: rs::ShadowDiff,
    elapsed_ms: f64,
}

impl Report {
    fn summary(&self) -> String {
        let d = &self.diff;
        format!(
            "[railgun] shadow G7 route={} launches={} fully_certified={}/{} ib_dwords redline={} railgun={} canonical_equal={} \
             nop_dwords redline={} railgun={} kernargs={}/{} bindings={}/{} words={}/{} boundaries equal={} stronger={} weaker={} \
             benign_relocation={} pacing_mismatch={} railgun_stronger={} railgun_weaker={} g7={} errors={} elapsed_ms={:.1}",
            self.route.as_deref().unwrap_or("unmatched"),
            self.tape.launches,
            self.facts.fully_certified,
            self.facts.launches,
            d.ib.redline_dwords,
            d.ib.railgun_dwords,
            d.ib.equal_canonical,
            d.ib.redline_nop_dwords,
            d.ib.railgun_nop_dwords,
            d.kernargs.equal,
            d.kernargs.compared,
            d.bindings.equal,
            d.bindings.compared,
            d.word_patches.equal,
            d.word_patches.compared,
            d.boundaries.equal,
            d.boundaries.stronger,
            d.boundaries.weaker,
            d.verdict.benign_relocation,
            d.verdict.pacing_mismatch,
            d.verdict.railgun_stronger,
            d.verdict.railgun_weaker,
            if d.verdict.g7_pass { "PASS" } else { "FAIL" },
            d.errors.len(),
            self.elapsed_ms,
        )
    }
}
