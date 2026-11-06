// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The per-arch cache table (design §1.3), hazard edges (§1.4) and the PM4
//! boundary plan of a kernel program (§2.1 segment rule, §2.2 per edge).
//!
//! Edges are derived from A1 effects at allocation granularity over the open
//! queue epoch (§1.4 rules 1, 3, 5): between a node and every earlier node of
//! its segment since the last completion wait. The gfx12 writer-side
//! retained-vector-line hazard that Redline encodes by name
//! (`requires_gfx12_pre_dispatch_vmem_acquire`, `mq_rotate_x` and
//! `fused_silu_mul_mq_rotate`) is the derived [`Hazard::WarPreWriter`] rule of
//! §1.3: a VMEM store into an allocation that an earlier node of the same
//! segment loaded through VMEM, with no system acquire in between.
//!
//! Each edge takes its row's visibility; a boundary emits the join of its
//! uncovered edges. An edge is covered by an earlier boundary of the segment
//! that already carries at least its visibility after the producer.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

use crate::coresident::GroupPlan;
use crate::kernel::{CondLowering, Conditional, KernelProgram, MemClass};
use crate::ResourceId;

/// `ACQUIRE_MEM` cache rungs railgun lowers to, ordered by what they
/// invalidate. The packet each rung lowers to per arch:
///
/// | rung | gfx12 | exact gfx1100 (gfx11 legacy PM4) |
/// |---|---|---|
/// | `InterNode` | `acquire_inter_node_gfx12`, GCR 0x10180 (scalar + vector L0) | `acquire_inter_node_same_agent`, GCR 0x380 (scalar + vector L0/GL1; no GL2 writeback) |
/// | `System` | `acquire_system_gfx12`, GCR 0xc3b1 (adds GL1 and GL2 writeback + invalidate) | ROCr's gfx10+ system `ACQUIRE_MEM` (`acquire_system`: GL2 writeback + invalidate) |
///
/// `WaitIdle` is `EVENT_WRITE CS_PARTIAL_FLUSH` on both families. The G4
/// probe names (`wait_inter_node`, `wait_system`) are the same on both: on
/// gfx11 the probe resolves them to the packets above.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Rung {
    #[default]
    None,
    InterNode,
    System,
}

/// §1.1 `Visibility` of a boundary before a node.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub struct Visibility {
    pub wait_idle: bool,
    pub acquire: Rung,
    /// The boundary exists for a gfx12 writer-side WAR (§1.3 row 2).
    pub pre_writer: bool,
}

impl Visibility {
    pub fn join(self, other: Self) -> Self {
        Self {
            wait_idle: self.wait_idle || other.wait_idle,
            acquire: self.acquire.max(other.acquire),
            pre_writer: self.pre_writer || other.pre_writer,
        }
    }

    pub fn is_empty(&self) -> bool {
        !self.wait_idle && self.acquire == Rung::None
    }

    /// `self` orders and invalidates at least what `need` asks for.
    pub fn covers(&self, need: &Self) -> bool {
        (self.wait_idle || !need.wait_idle) && self.acquire >= need.acquire
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Hazard {
    /// Producer stores, consumer loads (VMEM or SMEM).
    Raw,
    /// Consumer overwrites what a node of the open epoch reads.
    War,
    Waw,
    /// gfx12: VMEM store into an allocation a node of the same segment
    /// loaded through VMEM, with no system acquire since (§1.3 row 2).
    WarPreWriter,
}

/// One row of a per-arch cache table. `g4_receipt` is the silicon receipt
/// that would back the rung (G4); none exists yet, so every row the program
/// uses is an open obligation.
#[derive(Clone, Debug, Serialize)]
pub struct CacheRow {
    pub hazard: Hazard,
    pub visibility: Visibility,
    /// Why this rung is tier-D evidence today.
    pub evidence: &'static str,
    pub g4_receipt: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CacheTable {
    pub arch: String,
    pub rows: Vec<CacheRow>,
}

const GFX12_PARITY: &str = "Redline gfx1201 retained PM4 at WaitIdle + acquire_inter_node_gfx12 on every resource-dependent boundary: \
     dec-redline/land parity PASS at ctx 512/8k/32k, 200-replay stress, 33,000-step length sweep (perf/decode-1201/land/report.md §3, \
     decode-1201-r2/report.md §1); replay.rs:5349-5389";
const GFX12_PRE_WRITER: &str = "Redline's name-keyed writer-side system acquire before mq_rotate_x / fused_silu_mul_mq_rotate \
     (replay.rs:2987-2989, 5343-5376: a retained GC12 vector-cache line of the reused rotate destination), parity as above";
const GFX1100_PARITY: &str = "Redline gfx1100 retained PM4 (prepare_pm4_prefix_inner): CS_PARTIAL_FLUSH + acquire_inter_node_same_agent (GCR 0x380) \
     on every resource-dependent boundary (the HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE rung is opt-in, default off and not modelled); \
     tier-D evidence class only: no railgun G4 silicon receipt exists for gfx1100";
const GFX1100_PRE_WRITER: &str = "Redline gfx11 has NO writer-side pre-dispatch acquire (requires_gfx12_pre_dispatch_vmem_acquire is gfx12-only); \
     the tier-D row is a bare WaitIdle (G4 probe rung `wait`, the weakest rung that is safe for war_pre_writer on gfx1100 in the September run), \
     so a boundary joined with the dependent row is byte-identical to Redline's. Until it is receipted the strict policy \
     (RowPolicy::UnreceiptedSystem) lowers this hazard WaitIdle + System";

impl CacheTable {
    /// The table for `arch`, or `None` where railgun has no rows: gfx12
    /// (`gfx1200`, `gfx1201`) and exact `gfx1100`. Every other gfx11 part
    /// (gfx1101/1102/1150/1151/1152) has no rows: their command processors
    /// and caches are unverified here.
    pub fn for_arch(arch: &str) -> Option<Self> {
        if arch == "gfx1100" {
            return Some(Self::gfx1100());
        }
        if !matches!(arch, "gfx1200" | "gfx1201") {
            return None;
        }
        let dependent = Visibility {
            wait_idle: true,
            acquire: Rung::InterNode,
            pre_writer: false,
        };
        let row = |hazard, visibility, evidence| CacheRow {
            hazard,
            visibility,
            evidence,
            g4_receipt: None,
        };
        Some(Self {
            arch: arch.to_owned(),
            rows: vec![
                row(Hazard::Raw, dependent, GFX12_PARITY),
                row(Hazard::War, dependent, GFX12_PARITY),
                row(Hazard::Waw, dependent, GFX12_PARITY),
                row(
                    Hazard::WarPreWriter,
                    Visibility {
                        wait_idle: true,
                        acquire: Rung::System,
                        pre_writer: true,
                    },
                    GFX12_PRE_WRITER,
                ),
            ],
        })
    }

    /// Exact gfx1100 (RDNA3 Navi 31, 7900 XTX), tier-D: Redline's gfx11
    /// retained-PM4 policy (`CS_PARTIAL_FLUSH` + `acquire_inter_node_same_agent`
    /// per dependent boundary). The writer-side row is a bare `WaitIdle`
    /// (probe rung `wait`; the empty row would probe as `none`, which
    /// removes the WAR wait itself and is stale by construction): Redline
    /// gfx11 has no counterpart of the gfx12 pre-dispatch acquire. Joined
    /// with a WAR/WAW/RAW row it lowers byte-identically to Redline. A
    /// product MUST lower under [`RowPolicy::UnreceiptedSystem`] until
    /// [`Self::attach_receipts`] has attached a gfx1100 G4 probe table.
    fn gfx1100() -> Self {
        let dependent = Visibility {
            wait_idle: true,
            acquire: Rung::InterNode,
            pre_writer: false,
        };
        let row = |hazard, visibility, evidence| CacheRow {
            hazard,
            visibility,
            evidence,
            g4_receipt: None,
        };
        Self {
            arch: "gfx1100".to_owned(),
            rows: vec![
                row(Hazard::Raw, dependent, GFX1100_PARITY),
                row(Hazard::War, dependent, GFX1100_PARITY),
                row(Hazard::Waw, dependent, GFX1100_PARITY),
                row(
                    Hazard::WarPreWriter,
                    Visibility {
                        wait_idle: true,
                        acquire: Rung::None,
                        pre_writer: true,
                    },
                    GFX1100_PRE_WRITER,
                ),
            ],
        }
    }

    /// Attach G4 silicon receipts (design §5 G4) from a probe cache table
    /// (`cache-table.<arch>.json`, `examples/g4_probe`). A row is receipted
    /// when every probe row backing it ran the table's own rung for at least
    /// `min_trials` trials with no stale observation. A backing row that saw
    /// the table's rung stale fails the whole attach: the tier-D rung is
    /// unsafe on this silicon. Rows the probe did not cover stay open
    /// obligations. `source` names the file (path and digest) in each receipt.
    pub fn attach_receipts(
        &mut self,
        probe: &G4Table,
        source: &str,
        min_trials: u64,
    ) -> Result<usize, String> {
        if probe.identity.arch != self.arch {
            return Err(format!(
                "G4 table is for {}, cache table for {}",
                probe.identity.arch, self.arch
            ));
        }
        let mut attached = 0;
        for row in &mut self.rows {
            let rung = probe_rung(row.visibility);
            let backing: &[&str] = match row.hazard {
                Hazard::Raw => &["raw_vmem", "raw_smem"],
                Hazard::War => &["war"],
                Hazard::Waw => &["waw"],
                Hazard::WarPreWriter => &["war_pre_writer"],
            };
            let mut evidence = Vec::new();
            for name in backing {
                let Some(probed) = probe.rows.iter().find(|r| r.row == *name) else {
                    break;
                };
                let Some(result) = probed.trials.iter().find(|t| t.rung == rung) else {
                    break;
                };
                if result.stale_trials != 0 {
                    return Err(format!(
                        "{:?} row: probe row {name} saw {} stale trials at the table rung {rung} ({} trials)",
                        row.hazard, result.stale_trials, result.trials
                    ));
                }
                if result.trials < min_trials {
                    break;
                }
                evidence.push(format!(
                    "{name}@{rung}: {} trials, 0 stale; weakest safe {}",
                    result.trials,
                    probed.weakest_safe_rung.as_deref().unwrap_or("none clean")
                ));
            }
            if evidence.len() == backing.len() {
                row.g4_receipt = Some(format!("{source}: {}", evidence.join("; ")));
                attached += 1;
            }
        }
        Ok(attached)
    }

    pub fn row(&self, hazard: Hazard) -> &CacheRow {
        self.rows
            .iter()
            .find(|r| r.hazard == hazard)
            .expect("every hazard has a row")
    }
}

/// The probe rung name (`examples/g4_probe`) a table visibility lowers to.
pub fn probe_rung(v: Visibility) -> &'static str {
    match (v.wait_idle, v.acquire) {
        (false, Rung::None) => "none",
        (true, Rung::None) => "wait",
        (_, Rung::InterNode) => "wait_inter_node",
        (_, Rung::System) => "wait_system",
    }
}

/// Per-arch packet shaping of railgun's PM4 lowering (design §4.2): applied
/// uniformly to every dispatch of every segment lowered for the arch. It is
/// a property of the arch's command processor, never keyed on a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ArchLowering {
    /// Body dwords of one type-3 `NOP` emitted after every `DISPATCH_DIRECT`
    /// (the builder's `Gfx12DispatchPacing::PostDispatchNop`); 0 = none.
    pub post_dispatch_nop: u32,
}

/// gfx1201 post-dispatch NOP: the plateau of the NOP-length sweep at auto
/// clocks (32/64/128 body dwords −0.375/−0.424/−0.451 ms at ctx 512; placing
/// the pad *before* the dispatch costs time), shipped as Redline's
/// `PostDispatchNop(64)` (`0466dbd01`; `/home/kaden/qcal/perf/decode-1201-r2/report.md`
/// §2.2). NOPs write no register or memory, so the lowering stays bit-exact.
pub const GFX1201_POST_DISPATCH_NOP: u32 = 64;

impl ArchLowering {
    /// The lowering parameters for `arch`. Only exact gfx1201 is paced: the
    /// sweep ran there, and gfx1200 has no bench card. Exact gfx1100 is
    /// explicitly unpaced: the post-dispatch `NOP` is a gfx12 command
    /// processor property (`Gfx12DispatchPacing`; the gfx11 builder has no
    /// pacing packet), and nothing has measured one on gfx1100.
    pub fn for_arch(arch: &str) -> Self {
        match arch {
            "gfx1201" => Self {
                post_dispatch_nop: GFX1201_POST_DISPATCH_NOP,
            },
            "gfx1100" => Self {
                post_dispatch_nop: 0,
            },
            _ => Self {
                post_dispatch_nop: 0,
            },
        }
    }
}

/// #867: arches whose compute queue is verified on silicon to honour
/// `COND_EXEC` (PM4 0x22) for both a set and a clear flag. gfx1151 (GFX11.5)
/// only: other gfx11 parts are unverified and gfx12 compute `COND_EXEC` has
/// no hardware receipt, so their conditionals must use the kernel guard.
pub fn cp_skip_admitted(arch: &str) -> bool {
    arch == "gfx1151"
}

/// The parts of a G4 probe cache table (`cache-table.<arch>.json`) the
/// receipts use.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct G4Table {
    pub identity: G4Identity,
    pub rows: Vec<G4Row>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct G4Identity {
    pub arch: String,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct G4Row {
    pub row: String,
    pub weakest_safe_rung: Option<String>,
    pub trials: Vec<G4RungTrials>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct G4RungTrials {
    pub rung: String,
    pub trials: u64,
    pub stale_trials: u64,
}

/// How rows without a G4 receipt lower (§2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RowPolicy {
    /// Use the row's tier-D rung and record a G4 obligation (the M1 shadow
    /// comparison: tier D is Redline's evidence class, D13).
    TierD,
    /// §2.2 row 3 read strictly: no G4 receipt ⇒ `WaitIdle + System`.
    UnreceiptedSystem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Edge {
    pub producer: usize,
    pub consumer: usize,
    pub hazard: Hazard,
    pub resource: u32,
}

/// The boundary before one node.
#[derive(Clone, Debug, Serialize)]
pub struct Boundary {
    pub visibility: Visibility,
    /// Uncovered edges ending at this node (the reasons for `visibility`).
    pub edges: Vec<Edge>,
    /// Segment edge: the IB entry acquire, or a full barrier around a
    /// non-PM4 node (§1.4 rule 4).
    pub split: bool,
}

/// Consecutive PM4 nodes lowered to one IB.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Segment {
    pub first: usize,
    pub end: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Pm4Plan {
    pub policy: RowPolicy,
    pub segments: Vec<Segment>,
    /// One per node: the boundary emitted before it.
    pub boundaries: Vec<Boundary>,
    /// Boundaries whose visibility each row decided.
    pub rows_used: BTreeMap<Hazard, usize>,
    /// #867 `CpSkip` conditionals, in node order.
    pub skips: Vec<Conditional>,
    /// Co-resident groups ([`KernelProgram::add_coresident_group`]), in node
    /// order, with their member-member classification and, once admitted,
    /// their residency proof.
    pub groups: Vec<GroupPlan>,
    /// Why the program is not certified: its `LabPending` nodes and
    /// uncertified group orderings ([`Pm4Plan::admission`]; empty in a
    /// product build).
    pub pending: Vec<String>,
}

/// One PM4 command of a segment, before packet encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pm4Op {
    /// HIP→PM4 ownership acquire (system scope).
    Entry,
    Boundary(Visibility),
    Dispatch(usize),
    /// Terminal wait + system acquire (GL2 writeback for the next consumer).
    Exit,
    /// #867 CP skip opens: `RELEASE_MEM` + `WAIT_REG_MEM` on a lowering-owned
    /// fence word (every earlier dispatch done, the flag fresh), then
    /// `COND_EXEC` on the flag resource's first dword. Every command up to
    /// the matching [`Pm4Op::SkipEnd`] is discarded when it is zero; the
    /// lowering patches the count there and keeps no register state across
    /// either edge.
    SkipBegin(ResourceId),
    SkipEnd,
}

/// The HIP→PM4 ownership acquire before a segment's first dispatch.
pub const ENTRY: Visibility = Visibility {
    wait_idle: false,
    acquire: Rung::System,
    pre_writer: false,
};
/// Segment exit + drained HIP stream + entry: the full barrier around a non-PM4 node.
pub const BARRIER: Visibility = Visibility {
    wait_idle: true,
    acquire: Rung::System,
    pre_writer: false,
};

/// Derive edges and boundaries for `program` under `table`.
pub fn plan(program: &KernelProgram, table: &CacheTable, policy: RowPolicy) -> Pm4Plan {
    let visibility_of = |hazard: Hazard| {
        let row = table.row(hazard);
        match (policy, &row.g4_receipt) {
            (RowPolicy::TierD, _) | (RowPolicy::UnreceiptedSystem, Some(_)) => row.visibility,
            (RowPolicy::UnreceiptedSystem, None) => Visibility {
                wait_idle: true,
                acquire: Rung::System,
                pre_writer: row.visibility.pre_writer,
            },
        }
    };
    let n = program.nodes.len();
    let mut boundaries = Vec::with_capacity(n);
    let mut segments = Vec::new();
    let mut rows_used = BTreeMap::new();
    let mut open: Vec<usize> = Vec::new();
    let mut last_vmem_load: HashMap<ResourceId, usize> = HashMap::new();
    // Boundary position (node index) of the latest system acquire in the
    // segment: readers before it are covered for WarPreWriter.
    let mut last_system = 0usize;
    let mut segment_start: Option<usize> = None;
    let skips: Vec<Conditional> = program
        .conditionals
        .iter()
        .filter(|c| c.lowering == CondLowering::CpSkip)
        .cloned()
        .collect();
    // #867: a skipped body emits none of its boundaries, so coverage they
    // provide must not outlive the body. The epoch state at the body's
    // entry is kept and joined back at its end (a skipped body is treated as
    // executed: its own effects stay open).
    let mut entry_state: Option<(Vec<usize>, HashMap<ResourceId, usize>, usize)> = None;
    let mut open_group: Option<usize> = None;
    let group_plans: Vec<GroupPlan> = program
        .groups
        .iter()
        .enumerate()
        .map(|(index, g)| {
            let (admitted, refusals) = crate::coresident::classify(program, g);
            GroupPlan {
                nodes: g.nodes.clone(),
                admitted,
                refusals,
                residency: None,
                pending: {
                    let mut pending = crate::admission::node_pending(program, g.nodes.clone());
                    pending.extend(crate::admission::ordering_pending(index, g));
                    pending
                },
            }
        })
        .collect();
    for (j, node) in program.nodes.iter().enumerate() {
        if skips.iter().any(|c| c.body.end == j) {
            if let Some((was_open, loads, system)) = entry_state.take() {
                open.extend(was_open);
                open.sort_unstable();
                open.dedup();
                for (resource, i) in loads {
                    last_vmem_load.entry(resource).or_insert(i);
                }
                last_system = last_system.min(system);
            }
        }
        if skips.iter().any(|c| c.body.start == j) {
            entry_state = Some(if segment_start.is_some() {
                (open.clone(), last_vmem_load.clone(), last_system)
            } else {
                (Vec::new(), HashMap::new(), j)
            });
        }
        if !node.lowering.is_pm4_lowerable() {
            open_group = None;
            if let Some(first) = segment_start.take() {
                segments.push(Segment { first, end: j });
            }
            boundaries.push(Boundary {
                visibility: BARRIER,
                edges: Vec::new(),
                split: true,
            });
            continue;
        }
        let group_member = program.groups.iter().position(|g| g.nodes.contains(&j));
        // A later member of the group whose first member already emitted the
        // join of every member's pre-group edges: no boundary of its own
        // (members run concurrently; their mutual edges are classified in
        // `GroupPlan`, never answered by a wait).
        if let (Some(g), Some(open_g)) = (group_member, open_group) {
            if g == open_g && segment_start.is_some() {
                boundaries.push(Boundary {
                    visibility: Visibility::default(),
                    edges: Vec::new(),
                    split: false,
                });
                open.push(j);
                note_loads(program, j, &mut last_vmem_load);
                continue;
            }
        }
        let group_start = group_member.filter(|&g| program.groups[g].nodes.start == j);
        let Some(first) = segment_start else {
            segment_start = Some(j);
            open.clear();
            open.push(j);
            last_vmem_load.clear();
            last_system = j;
            open_group = group_start;
            // After a non-PM4 node the entry follows a drained HIP stream.
            let after_barrier = j > 0;
            let visibility = if after_barrier { BARRIER } else { ENTRY };
            boundaries.push(Boundary {
                visibility,
                edges: Vec::new(),
                split: true,
            });
            note_loads(program, j, &mut last_vmem_load);
            continue;
        };
        debug_assert!(first < j);
        // The first member of a group answers for every member: all of them
        // run after this boundary, against the nodes that precede the group.
        let consumers = match group_start {
            Some(g) => program.groups[g].nodes.clone(),
            None => j..j + 1,
        };
        let mut edges = Vec::new();
        for m in consumers {
            for effect in &program.nodes[m].effects {
                for &i in &open {
                    for earlier in &program.nodes[i].effects {
                        if earlier.resource != effect.resource {
                            continue;
                        }
                        let mut push = |hazard| {
                            edges.push(Edge {
                                producer: i,
                                consumer: m,
                                hazard,
                                resource: effect.resource.0,
                            })
                        };
                        if earlier.writes() && effect.reads() {
                            push(Hazard::Raw);
                        }
                        if earlier.reads() && effect.writes() {
                            push(Hazard::War);
                        }
                        if earlier.writes() && effect.writes() {
                            push(Hazard::Waw);
                        }
                    }
                }
                if effect.has(MemClass::VmemStore) {
                    if let Some(&i) = last_vmem_load.get(&effect.resource) {
                        if i >= last_system && i < j {
                            edges.push(Edge {
                                producer: i,
                                consumer: m,
                                hazard: Hazard::WarPreWriter,
                                resource: effect.resource.0,
                            });
                        }
                    }
                }
            }
        }
        edges.sort_by_key(|e| (e.hazard, e.producer, e.consumer, e.resource));
        edges.dedup();
        let visibility = edges.iter().fold(Visibility::default(), |acc, e| {
            acc.join(visibility_of(e.hazard))
        });
        for hazard in edges
            .iter()
            .map(|e| e.hazard)
            .collect::<std::collections::BTreeSet<_>>()
        {
            *rows_used.entry(hazard).or_insert(0usize) += 1;
        }
        if visibility.wait_idle {
            open.clear();
        }
        open.push(j);
        if visibility.acquire == Rung::System {
            last_system = j;
        }
        note_loads(program, j, &mut last_vmem_load);
        open_group = group_start;
        boundaries.push(Boundary {
            visibility,
            edges,
            split: false,
        });
    }
    if let Some(first) = segment_start {
        segments.push(Segment { first, end: n });
    }
    Pm4Plan {
        policy,
        segments,
        boundaries,
        rows_used,
        skips,
        groups: group_plans,
        pending: crate::admission::program_pending(program),
    }
}

fn note_loads(
    program: &KernelProgram,
    node: usize,
    last_vmem_load: &mut HashMap<ResourceId, usize>,
) {
    for effect in &program.nodes[node].effects {
        if effect.has(MemClass::VmemLoad) {
            last_vmem_load.insert(effect.resource, node);
        }
    }
}

impl Pm4Plan {
    /// The command sequence of one segment. A `CpSkip` body (all PM4, so
    /// inside one segment) is bracketed by `SkipBegin`/`SkipEnd`, with its
    /// first node's boundary inside the skipped range.
    pub fn segment_ops(&self, segment: &Segment) -> Vec<Pm4Op> {
        let mut ops = Vec::with_capacity(2 * (segment.end - segment.first) + 2);
        ops.push(Pm4Op::Entry);
        for node in segment.first..segment.end {
            if let Some(c) = self.skips.iter().find(|c| c.body.start == node) {
                ops.push(Pm4Op::SkipBegin(c.flag));
            }
            let b = &self.boundaries[node];
            if node != segment.first && !b.visibility.is_empty() {
                ops.push(Pm4Op::Boundary(b.visibility));
            }
            ops.push(Pm4Op::Dispatch(node));
            if self.skips.iter().any(|c| c.body.end == node + 1) {
                ops.push(Pm4Op::SkipEnd);
            }
        }
        ops.push(Pm4Op::Exit);
        ops
    }

    /// Mid-segment boundaries with a wait and with each rung.
    pub fn counts(&self) -> PlanCounts {
        let mut c = PlanCounts::default();
        for b in self.boundaries.iter().filter(|b| !b.split) {
            c.waits += usize::from(b.visibility.wait_idle);
            c.inter_node += usize::from(b.visibility.acquire == Rung::InterNode);
            c.system += usize::from(b.visibility.acquire == Rung::System);
            c.pre_writer += usize::from(b.visibility.pre_writer);
            c.elided += usize::from(b.visibility.is_empty());
        }
        c.segments = self.segments.len();
        c
    }

    /// True when one segment spans nodes `0..nodes` and nothing else: the
    /// "one segment" shape of a `Whole16` window (design A §2, slice A1).
    pub fn is_single_segment(&self, nodes: usize) -> bool {
        matches!(
            self.segments.as_slice(),
            [only] if only.first == 0 && only.end == nodes
        )
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PlanCounts {
    pub segments: usize,
    pub waits: usize,
    pub inter_node: usize,
    pub system: usize,
    pub pre_writer: usize,
    pub elided: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::tests::{launch, ptr};
    use crate::kernel::{author, MemClass::*};

    /// A chain over allocations A (0x7000_0000) and B (0x7100_0000):
    /// n0 loads A, stores B; n1 loads B, stores C; n2 stores A (writer of an
    /// allocation n0 loaded); n3 loads D only (independent of n2).
    fn chain() -> crate::kernel::KernelProgram {
        let (a, b, c, d) = (
            0x7000_0000u64,
            0x7100_0000u64,
            0x7200_0000u64,
            0x7300_0000u64,
        );
        let two = vec![ptr(0, &[VmemLoad]), ptr(8, &[VmemStore])];
        let launches = vec![
            launch(&[(0, a), (8, b)], 16, two.clone()),
            launch(&[(0, b), (8, c)], 16, two.clone()),
            launch(&[(8, a)], 16, vec![ptr(8, &[VmemStore])]),
            launch(&[(0, d)], 16, vec![ptr(0, &[VmemLoad])]),
        ];
        author("gfx1201", &launches)
    }

    #[test]
    fn raw_takes_the_dependent_row_and_independent_nodes_elide() {
        let table = CacheTable::for_arch("gfx1201").unwrap();
        let p = plan(&chain(), &table, RowPolicy::TierD);
        assert_eq!(p.segments.len(), 1);
        assert_eq!(p.boundaries[0].visibility, ENTRY);
        assert_eq!(
            p.boundaries[1].visibility,
            Visibility {
                wait_idle: true,
                acquire: Rung::InterNode,
                pre_writer: false
            }
        );
        assert!(p.boundaries[1]
            .edges
            .iter()
            .all(|e| e.hazard == Hazard::Raw && e.producer == 0));
        assert!(
            p.boundaries[3].visibility.is_empty(),
            "n3 touches only D: no edge, no wait"
        );
    }

    #[test]
    fn writer_after_an_earlier_vmem_load_gets_the_system_pre_writer_rung() {
        let table = CacheTable::for_arch("gfx1201").unwrap();
        let p = plan(&chain(), &table, RowPolicy::TierD);
        // n2 stores A, loaded by n0 two nodes earlier: n0 is outside the open
        // epoch (n1's wait closed it) but no system acquire intervened.
        let b = &p.boundaries[2];
        assert_eq!(
            b.visibility,
            Visibility {
                wait_idle: true,
                acquire: Rung::System,
                pre_writer: true
            }
        );
        assert_eq!(
            b.edges
                .iter()
                .map(|e| (e.hazard, e.producer))
                .collect::<Vec<_>>(),
            [(Hazard::WarPreWriter, 0)]
        );
    }

    #[test]
    fn a_system_boundary_covers_earlier_readers_for_later_writers() {
        let a = 0x7000_0000u64;
        let launches = vec![
            launch(&[(0, a)], 16, vec![ptr(0, &[VmemLoad])]),
            launch(&[(8, a)], 16, vec![ptr(8, &[VmemStore])]),
            launch(&[(8, a)], 16, vec![ptr(8, &[VmemStore])]),
        ];
        let program = author("gfx1201", &launches);
        let p = plan(
            &program,
            &CacheTable::for_arch("gfx1201").unwrap(),
            RowPolicy::TierD,
        );
        // n1 overwrites what n0 loaded: WAR + pre-writer ⇒ system acquire.
        assert_eq!(
            p.boundaries[1].visibility,
            Visibility {
                wait_idle: true,
                acquire: Rung::System,
                pre_writer: true
            }
        );
        // n2 overwrites it again: n0's load is behind n1's system acquire, so
        // only the WAW on n1 remains.
        assert_eq!(
            p.boundaries[2].visibility,
            Visibility {
                wait_idle: true,
                acquire: Rung::InterNode,
                pre_writer: false
            }
        );
        assert_eq!(
            p.boundaries[2]
                .edges
                .iter()
                .map(|e| e.hazard)
                .collect::<Vec<_>>(),
            [Hazard::Waw]
        );
        // §2.2 read strictly (no G4 receipt ⇒ system) lifts the WAW too.
        let strict = plan(
            &program,
            &CacheTable::for_arch("gfx1201").unwrap(),
            RowPolicy::UnreceiptedSystem,
        );
        assert_eq!(strict.boundaries[2].visibility.acquire, Rung::System);
    }

    fn chain_gfx1100() -> crate::kernel::KernelProgram {
        let mut p = chain();
        p.arch = "gfx1100".into();
        p
    }

    #[test]
    fn gfx1100_table_is_exact_and_mirrors_redline_gfx11() {
        let t = CacheTable::for_arch("gfx1100").expect("gfx1100 table");
        assert_eq!(t.arch, "gfx1100");
        let dependent = Visibility {
            wait_idle: true,
            acquire: Rung::InterNode,
            pre_writer: false,
        };
        for hazard in [Hazard::Raw, Hazard::War, Hazard::Waw] {
            assert_eq!(t.row(hazard).visibility, dependent, "{hazard:?}");
            assert!(t.row(hazard).evidence.contains("same_agent"));
            assert!(t.row(hazard).g4_receipt.is_none());
        }
        // The writer-side row is a bare WaitIdle: probe rung `wait`. (The
        // empty row would probe as `none`, which removes the WAR wait itself
        // and is stale by construction.)
        assert_eq!(
            t.row(Hazard::WarPreWriter).visibility,
            Visibility {
                wait_idle: true,
                acquire: Rung::None,
                pre_writer: true
            }
        );
        assert_eq!(probe_rung(t.row(Hazard::WarPreWriter).visibility), "wait");
        assert!(t.row(Hazard::WarPreWriter).g4_receipt.is_none());
        // Exact device only: no other gfx11 part is a table.
        for other in [
            "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx11", "GFX1100",
            "gfx1100 ", "",
        ] {
            assert!(CacheTable::for_arch(other).is_none(), "{other:?}");
        }
        // The gfx12 table is untouched.
        assert_eq!(
            CacheTable::for_arch("gfx1201")
                .unwrap()
                .row(Hazard::WarPreWriter)
                .visibility,
            Visibility {
                wait_idle: true,
                acquire: Rung::System,
                pre_writer: true
            }
        );
    }

    #[test]
    fn gfx1100_tier_d_plan_has_no_gfx12_pre_writer_system_acquire() {
        let table = CacheTable::for_arch("gfx1100").unwrap();
        let p = plan(&chain_gfx1100(), &table, RowPolicy::TierD);
        assert!(p.is_single_segment(4));
        assert_eq!(p.boundaries[0].visibility, ENTRY);
        assert_eq!(
            p.boundaries[1].visibility,
            Visibility {
                wait_idle: true,
                acquire: Rung::InterNode,
                pre_writer: false
            }
        );
        // n2 stores A (n0 loaded it): the derived pre-writer edge's gfx11
        // tier-D row is a bare WaitIdle (probe rung `wait`), unlike gfx1201's
        // System row: no gfx12 system acquire on gfx11.
        assert_eq!(
            p.boundaries[2]
                .edges
                .iter()
                .map(|e| (e.hazard, e.producer))
                .collect::<Vec<_>>(),
            [(Hazard::WarPreWriter, 0)]
        );
        assert_eq!(
            p.boundaries[2].visibility,
            Visibility {
                wait_idle: true,
                acquire: Rung::None,
                pre_writer: true
            }
        );
        assert!(p.boundaries[3].visibility.is_empty());
        let c = p.counts();
        assert_eq!(
            (c.waits, c.inter_node, c.system, c.pre_writer),
            (2, 1, 0, 1)
        );
        // Joined with a WAR/WAW/RAW row the boundary is exactly Redline's
        // dependent boundary (wait + same-agent acquire).
        assert_eq!(
            table
                .row(Hazard::War)
                .visibility
                .join(table.row(Hazard::WarPreWriter).visibility)
                .acquire,
            Rung::InterNode
        );
    }

    #[test]
    fn gfx1100_unreceipted_rows_lower_wait_idle_plus_system() {
        let table = CacheTable::for_arch("gfx1100").unwrap();
        let p = plan(&chain_gfx1100(), &table, RowPolicy::UnreceiptedSystem);
        let system = Visibility {
            wait_idle: true,
            acquire: Rung::System,
            pre_writer: false,
        };
        // Every row without a receipt is WaitIdle + the strongest acquire.
        assert_eq!(p.boundaries[1].visibility, system);
        // n1's System acquire is behind n0's load, so n2's store to A raises
        // no pre-writer edge at all (the same coverage rule as gfx1201).
        assert!(p.boundaries[2].edges.is_empty());
        assert!(p.boundaries[2].visibility.is_empty());
        assert!(p.boundaries[3].visibility.is_empty());
        // A store into an allocation loaded earlier in the same open epoch
        // is a WAR edge on its own row: unreceipted, hence System.
        let a = 0x7000_0000u64;
        let launches = vec![
            launch(&[(0, a)], 16, vec![ptr(0, &[VmemLoad])]),
            launch(&[(8, a)], 16, vec![ptr(8, &[VmemStore])]),
        ];
        let program = author("gfx1100", &launches);
        let p = plan(&program, &table, RowPolicy::UnreceiptedSystem);
        // WaitIdle + System, flagged as a writer-side boundary (the pre-writer
        // row's own flag survives the strict policy).
        assert_eq!(
            p.boundaries[1].visibility,
            Visibility {
                pre_writer: true,
                ..system
            }
        );
        assert_eq!(
            p.boundaries[1]
                .edges
                .iter()
                .map(|e| e.hazard)
                .collect::<Vec<_>>(),
            [Hazard::War, Hazard::WarPreWriter]
        );
    }

    #[test]
    fn gfx1100_receipts_attach_against_the_gfx1100_probe_table() {
        const M: u64 = 1 << 20;
        let clean: &[(&str, u64, u64)] = &[("wait_inter_node", M, 0), ("wait", M, 0)];
        let probe = |arch: &str| {
            let mut g = g4(&[
                ("raw_vmem", clean),
                ("raw_smem", clean),
                ("war", clean),
                ("waw", clean),
                ("war_pre_writer", clean),
            ]);
            g.identity.arch = arch.into();
            g
        };
        // A gfx1201 probe never receipts the gfx1100 table.
        let mut t = CacheTable::for_arch("gfx1100").unwrap();
        assert!(t.attach_receipts(&probe("gfx1201"), "p", M).is_err());
        // A clean gfx1100 probe receipts all four rows (the pre-writer row
        // against the probe's `wait` rung), after which the strict policy
        // keeps each row's own visibility.
        assert_eq!(t.attach_receipts(&probe("gfx1100"), "p", M), Ok(4));
        let p = plan(&chain_gfx1100(), &t, RowPolicy::UnreceiptedSystem);
        assert_eq!(
            p.boundaries[1].visibility,
            Visibility {
                wait_idle: true,
                acquire: Rung::InterNode,
                pre_writer: false
            }
        );
        assert_eq!(
            p.boundaries[2].visibility,
            Visibility {
                wait_idle: true,
                acquire: Rung::None,
                pre_writer: true
            }
        );
        // A stale `wait` rung on the pre-writer probe row refuses the attach.
        let mut t = CacheTable::for_arch("gfx1100").unwrap();
        let mut stale = probe("gfx1100");
        stale
            .rows
            .iter_mut()
            .find(|r| r.row == "war_pre_writer")
            .unwrap()
            .trials
            .iter_mut()
            .find(|x| x.rung == "wait")
            .unwrap()
            .stale_trials = 5;
        assert!(t.attach_receipts(&stale, "p", M).is_err());
    }

    #[test]
    fn gfx1100_is_unpaced_and_cp_skip_is_refused() {
        assert_eq!(ArchLowering::for_arch("gfx1100").post_dispatch_nop, 0);
        assert_eq!(ArchLowering::for_arch("gfx1201").post_dispatch_nop, 64);
        assert!(!cp_skip_admitted("gfx1100"));
        assert!(!cp_skip_admitted("gfx1201"));
        assert!(cp_skip_admitted("gfx1151"));
    }

    fn g4(rows: &[(&str, &[(&str, u64, u64)])]) -> G4Table {
        G4Table {
            identity: G4Identity {
                arch: "gfx1201".into(),
            },
            rows: rows
                .iter()
                .map(|(row, rungs)| G4Row {
                    row: (*row).into(),
                    weakest_safe_rung: None,
                    trials: rungs
                        .iter()
                        .map(|(r, t, s)| G4RungTrials {
                            rung: (*r).into(),
                            trials: *t,
                            stale_trials: *s,
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    #[test]
    fn receipts_need_every_backing_row_clean_at_the_table_rung() {
        const M: u64 = 1 << 20;
        let clean = [
            ("wait", M, 0),
            ("wait_inter_node", M, 0),
            ("wait_system", M, 0),
        ];
        // RAW is backed by both the VMEM and the SMEM consumer rows.
        let mut table = CacheTable::for_arch("gfx1201").unwrap();
        let only_vmem = g4(&[
            ("raw_vmem", &clean),
            ("war", &clean),
            ("waw", &clean),
            ("war_pre_writer", &clean),
        ]);
        assert_eq!(table.attach_receipts(&only_vmem, "t", M), Ok(3));
        assert!(table.row(Hazard::Raw).g4_receipt.is_none());
        assert!(table.row(Hazard::WarPreWriter).g4_receipt.is_some());
        // Too few trials: no receipt, no error.
        let mut table = CacheTable::for_arch("gfx1201").unwrap();
        let short = g4(&[("waw", &[("wait_inter_node", M - 1, 0)])]);
        assert_eq!(table.attach_receipts(&short, "t", M), Ok(0));
        // The table rung itself seen stale: refused.
        let mut table = CacheTable::for_arch("gfx1201").unwrap();
        let stale = g4(&[
            ("raw_vmem", &clean),
            ("raw_smem", &[("wait_inter_node", M, 3)]),
        ]);
        assert!(table
            .attach_receipts(&stale, "t", M)
            .unwrap_err()
            .contains("raw_smem"));
        // A weaker rung being stale does not matter for the table rung.
        let mut table = CacheTable::for_arch("gfx1201").unwrap();
        let weaker_stale = g4(&[
            ("raw_vmem", &[("wait", M, 9), ("wait_inter_node", M, 0)]),
            ("raw_smem", &clean),
        ]);
        assert_eq!(table.attach_receipts(&weaker_stale, "t", M), Ok(1));
        assert!(table
            .row(Hazard::Raw)
            .g4_receipt
            .as_deref()
            .unwrap()
            .contains("raw_smem@wait_inter_node"));
    }

    #[test]
    fn a_non_pm4_node_splits_the_segment_with_full_barriers() {
        let mut launches = vec![launch(&[(0, 0x7000_0000)], 16, vec![ptr(0, &[VmemLoad])]); 3];
        launches[1].facts = Err("no record".into());
        let program = author("gfx1201", &launches);
        let p = plan(
            &program,
            &CacheTable::for_arch("gfx1201").unwrap(),
            RowPolicy::TierD,
        );
        assert_eq!(p.segments.len(), 2);
        assert!(p.boundaries[1].split && p.boundaries[1].visibility == BARRIER);
        assert!(p.boundaries[2].split && p.boundaries[2].visibility == BARRIER);
        assert_eq!(
            p.segment_ops(&p.segments[1]),
            [Pm4Op::Entry, Pm4Op::Dispatch(2), Pm4Op::Exit]
        );
    }

    /// n0 stores A; n1 (the body, flag C) loads A, so its boundary waits and
    /// closes n0's epoch, and stores B; n2 stores A. Executed, n2 orders only
    /// against n1's load (WAR). Skipped, n1's boundary never ran: n2 must
    /// still order against n0 (WAW).
    #[test]
    fn a_cp_skipped_body_covers_no_edge_that_crosses_it() {
        let (a, b, c) = (0x7000_0000u64, 0x7100_0000u64, 0x7200_0000u64);
        let launches = vec![
            launch(&[(8, a)], 16, vec![ptr(8, &[VmemStore])]),
            launch(
                &[(0, a), (8, b), (16, c)],
                24,
                vec![
                    ptr(0, &[VmemLoad]),
                    ptr(8, &[VmemStore]),
                    ptr(16, &[VmemLoad]),
                ],
            ),
            launch(&[(8, a)], 16, vec![ptr(8, &[VmemStore])]),
        ];
        let table = CacheTable::for_arch("gfx1201").unwrap();
        let mut program = author("gfx1151", &launches);
        let plain = plan(&program, &table, RowPolicy::TierD);
        let flag = program.nodes[1].effects[2].resource;
        program
            .add_conditional(Conditional {
                flag,
                body: 1..2,
                lowering: CondLowering::CpSkip,
            })
            .unwrap();
        let skipped = plan(&program, &table, RowPolicy::TierD);
        let producers = |p: &Pm4Plan| {
            let mut v: Vec<_> = p.boundaries[2]
                .edges
                .iter()
                .map(|e| (e.hazard, e.producer))
                .collect();
            v.sort();
            v
        };
        assert!(!producers(&plain).contains(&(Hazard::Waw, 0)));
        assert!(producers(&skipped).contains(&(Hazard::Waw, 0)));
        assert!(
            producers(&skipped).contains(&(Hazard::War, 1)),
            "the body stays executed"
        );
        assert_eq!(
            skipped.segment_ops(&skipped.segments[0]),
            [
                Pm4Op::Entry,
                Pm4Op::Dispatch(0),
                Pm4Op::SkipBegin(flag),
                Pm4Op::Boundary(skipped.boundaries[1].visibility),
                Pm4Op::Dispatch(1),
                Pm4Op::SkipEnd,
                Pm4Op::Boundary(skipped.boundaries[2].visibility),
                Pm4Op::Dispatch(2),
                Pm4Op::Exit,
            ]
        );
    }
}
