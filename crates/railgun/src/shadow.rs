//! G7 normalized route diff in shadow mode (design §5 G7, §7 M1): railgun
//! prepares its lowering of the same recorded tape next to Redline's and the
//! two are compared; railgun submits nothing.
//!
//! Surfaces: IB dwords (with the kernarg-buffer addresses in
//! `COMPUTE_USER_DATA_0/1` canonicalised to the dispatch index), loader
//! kernarg images, pointer bindings, word (kernarg) patch lists, grid patch
//! lists, and the per-boundary `(wait, acquire rung)` decoded from each IB.
//! Every difference is classified:
//! - `benign_relocation`: the bytes a replay runs are the same; only where a
//!   pointer lives or who re-encodes it differs;
//! - `railgun_stronger`: railgun orders or invalidates more, splits a
//!   segment, or patches more — allowed, counted, gated by G6;
//! - `railgun_weaker`: railgun waits or invalidates less than Redline at a
//!   boundary, keeps something constant that Redline patches, or disagrees on
//!   what runs. G7 fails on any of these unless a G4 receipt backs the row;
//!   no G4 receipt exists, so every one fails and is listed;
//! - `pacing_mismatch`: the post-dispatch `NOP` body dwords differ (the
//!   per-arch pacing of `ArchLowering`). Bit-exact either way, but it changes
//!   the tape's timing, so G7 requires railgun's pacing to equal Redline's.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::kernel::{Arg, KernelProgram, Lowering, WordRole};
use crate::plan::{Edge, Hazard, Pm4Plan, Rung, Visibility, ENTRY};

const PACKET3_NOP: u32 = 0x10;
const PACKET3_DISPATCH_DIRECT: u32 = 0x15;
const PACKET3_EVENT_WRITE: u32 = 0x46;
const PACKET3_ACQUIRE_MEM: u32 = 0x58;
const PACKET3_SET_SH_REG: u32 = 0x76;
const COMPUTE_USER_DATA_0: u32 = 0x240;
/// Written once per command buffer by the static-stateful builder.
const COMPUTE_RESOURCE_LIMITS: u32 = 0x215;
const CS_PARTIAL_FLUSH: u32 = 0x407;
/// `acquire_inter_node_gfx12` (GLK | GLV | SEQ_FORWARD).
const GCR_INTER_NODE: u32 = 0x1_0180;
/// `acquire_system_gfx12`.
const GCR_SYSTEM_GFX12: u32 = 0xc3b1;
/// `acquire_system` (the untrimmed ROCr template).
const GCR_SYSTEM_TEMPLATE: u32 = 0x1_c3f1;
/// `HipLlvmVmemL1` (GLV | GL1).
const GCR_VMEM_L1: u32 = 0x300;

/// A decoded `ACQUIRE_MEM` rung.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Acquire {
    None,
    Vmem,
    InterNode,
    System,
    Other(u32),
}

impl Acquire {
    fn from_gcr(gcr: u32) -> Self {
        match gcr {
            GCR_INTER_NODE => Self::InterNode,
            GCR_SYSTEM_GFX12 | GCR_SYSTEM_TEMPLATE => Self::System,
            GCR_VMEM_L1 => Self::Vmem,
            other => Self::Other(other),
        }
    }

    /// `self` invalidates at least what `other` does. System covers every
    /// same-agent rung; Vmem (GLV | GL1) and InterNode (GLK | GLV) are
    /// incomparable.
    pub fn covers(self, other: Self) -> bool {
        match (self, other) {
            (_, Self::None) => true,
            (a, b) if a == b => true,
            (Self::System, Self::Vmem | Self::InterNode) => true,
            _ => false,
        }
    }

    fn join(self, other: Self) -> Self {
        if self.covers(other) {
            self
        } else if other.covers(self) {
            other
        } else {
            Self::System
        }
    }
}

impl From<Rung> for Acquire {
    fn from(rung: Rung) -> Self {
        match rung {
            Rung::None => Self::None,
            Rung::InterNode => Self::InterNode,
            Rung::System => Self::System,
        }
    }
}

/// What an IB does before one dispatch (or at its end).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub struct BoundaryState {
    pub wait_idle: bool,
    pub acquire: Option<Acquire>,
    /// `NOP` body dwords (pacing).
    pub nop_dwords: u32,
}

impl BoundaryState {
    fn acquire(&self) -> Acquire {
        self.acquire.unwrap_or(Acquire::None)
    }

    /// `self` waits and invalidates at least what `other` does.
    pub fn covers(&self, other: &Self) -> bool {
        (self.wait_idle || !other.wait_idle) && self.acquire().covers(other.acquire())
    }

    fn of(v: Visibility) -> Self {
        Self { wait_idle: v.wait_idle, acquire: (v.acquire != Rung::None).then(|| v.acquire.into()), nop_dwords: 0 }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Packet {
    opcode: u32,
    dwords: Vec<u32>,
}

#[derive(Clone, Debug)]
struct Group {
    boundary: Vec<Packet>,
    /// SET_SH_REG packets and the DISPATCH_DIRECT.
    state: Vec<Packet>,
}

#[derive(Clone, Debug)]
struct Parsed {
    groups: Vec<Group>,
    epilogue: Vec<Packet>,
}

fn parse(ib: &[u32]) -> Result<Parsed, String> {
    let mut groups = Vec::new();
    let mut boundary = Vec::new();
    let mut state = Vec::new();
    let mut cursor = 0usize;
    while cursor < ib.len() {
        let header = ib[cursor];
        if header >> 30 != 3 {
            return Err(format!("dword {cursor}: not a type-3 packet header ({header:#010x})"));
        }
        let body = ((header >> 16) & 0x3fff) as usize + 1;
        let end = cursor + 1 + body;
        if end > ib.len() {
            return Err(format!("dword {cursor}: packet runs past the IB"));
        }
        let packet = Packet { opcode: (header >> 8) & 0xff, dwords: ib[cursor..end].to_vec() };
        cursor = end;
        match packet.opcode {
            PACKET3_SET_SH_REG => state.push(packet),
            PACKET3_DISPATCH_DIRECT => {
                state.push(packet);
                groups.push(Group { boundary: std::mem::take(&mut boundary), state: std::mem::take(&mut state) });
            }
            _ if !state.is_empty() => {
                return Err(format!("dword {}: opcode {:#x} between register writes and their dispatch", cursor - body - 1, packet.opcode));
            }
            _ => boundary.push(packet),
        }
    }
    if !state.is_empty() {
        return Err("register writes after the last dispatch".into());
    }
    Ok(Parsed { groups, epilogue: boundary })
}

fn decode_boundary(packets: &[Packet]) -> (BoundaryState, Vec<u32>) {
    let mut state = BoundaryState::default();
    let mut other = Vec::new();
    for p in packets {
        match p.opcode {
            PACKET3_EVENT_WRITE if p.dwords.get(1) == Some(&CS_PARTIAL_FLUSH) => state.wait_idle = true,
            PACKET3_ACQUIRE_MEM => {
                let gcr = *p.dwords.last().expect("packet has a body");
                let rung = Acquire::from_gcr(gcr);
                state.acquire = Some(state.acquire.map_or(rung, |a| a.join(rung)));
            }
            PACKET3_NOP => state.nop_dwords += p.dwords.len() as u32 - 1,
            opcode => other.push(opcode),
        }
    }
    (state, other)
}

/// The state packets with the kernarg address (`COMPUTE_USER_DATA_0/1`)
/// replaced by zero, and whether an address was found.
fn canonical_state(packets: &[Packet]) -> (Vec<u32>, Option<u64>) {
    let mut out = Vec::new();
    let mut address = None;
    for p in packets {
        let mut dwords = p.dwords.clone();
        if p.opcode == PACKET3_SET_SH_REG && dwords.len() >= 3 {
            let first = dwords[1];
            let values = dwords.len() - 2;
            // Queue-invariant registers are written once per command buffer
            // (`new_static_stateful`); a segment that starts mid-tape writes
            // them again with the same value.
            if first == COMPUTE_RESOURCE_LIMITS && values == 1 {
                continue;
            }
            if first <= COMPUTE_USER_DATA_0 && COMPUTE_USER_DATA_0 + 1 < first + values as u32 {
                let at = 2 + (COMPUTE_USER_DATA_0 - first) as usize;
                address = Some(u64::from(dwords[at]) | (u64::from(dwords[at + 1]) << 32));
                dwords[at] = 0;
                dwords[at + 1] = 0;
            }
        }
        out.extend(dwords);
    }
    (out, address)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Class {
    BenignRelocation,
    PacingMismatch,
    RailgunStronger,
    RailgunWeaker,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Surface {
    Ib,
    Kernarg,
    Binding,
    WordPatch,
    GridPatch,
    Boundary,
}

/// One kind of difference and every dispatch it occurs at.
#[derive(Clone, Debug, Serialize)]
pub struct Difference {
    pub class: Class,
    pub surface: Surface,
    pub detail: String,
    pub count: usize,
    pub dispatches: Vec<usize>,
}

/// Why Redline placed its boundary (its own tables), for the report.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct RedlineDecision {
    /// Where Redline's effects for this launch came from.
    pub effects: String,
    pub resource_independent: bool,
    /// `required_mid_acquire` (or the configured mid-acquire table) fired.
    pub name_acquire: bool,
    /// `requires_gfx12_pre_dispatch_vmem_acquire` fired.
    pub pre_dispatch_name: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct SlotBinding {
    pub offset: u32,
    pub base: u64,
    pub bytes: u64,
    pub interior: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct WordPatch {
    pub dispatch: usize,
    pub offset: u32,
    pub kind: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct GridPatch {
    pub dispatch: usize,
    pub axis: u8,
    pub addend: u32,
    pub divisor: u32,
}

#[derive(Clone, Debug)]
pub struct RedlineDispatch {
    pub symbol: String,
    /// The prepared loader kernarg image.
    pub kernarg: Vec<u8>,
    pub kernarg_address: u64,
    pub bindings: Vec<SlotBinding>,
    pub decision: RedlineDecision,
}

/// Redline's prepared single-IB tape.
#[derive(Clone, Debug)]
pub struct RedlineTape {
    pub ib: Vec<u32>,
    pub dispatches: Vec<RedlineDispatch>,
    pub word_patches: Vec<WordPatch>,
    pub grid_patches: Vec<GridPatch>,
}

/// One railgun segment's command dwords.
#[derive(Clone, Debug)]
pub struct SegmentIb {
    pub first: usize,
    pub end: usize,
    pub ib: Vec<u32>,
}

/// railgun's prepared (not submitted) PM4 lowering.
#[derive(Clone, Debug)]
pub struct RailgunLowering {
    pub segments: Vec<SegmentIb>,
    /// Per node: the kernarg image and its buffer address (PM4 nodes only).
    pub kernargs: Vec<Option<(Vec<u8>, u64)>>,
    /// `ArchLowering::post_dispatch_nop` the segments were lowered with.
    pub post_dispatch_nop: u32,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Tally {
    pub waits: usize,
    pub inter_node: usize,
    pub system: usize,
    pub vmem: usize,
    pub other: usize,
    pub elided: usize,
    pub nop_dwords: u64,
}

impl Tally {
    fn add(&mut self, b: &BoundaryState) {
        self.waits += usize::from(b.wait_idle);
        match b.acquire() {
            Acquire::None => {}
            Acquire::InterNode => self.inter_node += 1,
            Acquire::System => self.system += 1,
            Acquire::Vmem => self.vmem += 1,
            Acquire::Other(_) => self.other += 1,
        }
        self.elided += usize::from(!b.wait_idle && b.acquire() == Acquire::None);
        self.nop_dwords += u64::from(b.nop_dwords);
    }
}

/// A boundary where railgun is weaker than Redline: G7 fails here until a G4
/// receipt backs the weaker rung.
#[derive(Clone, Debug, Serialize)]
pub struct WeakBoundary {
    pub dispatch: usize,
    pub previous: String,
    pub symbol: String,
    pub redline: BoundaryState,
    pub railgun: BoundaryState,
    pub railgun_edges: Vec<Edge>,
    pub redline_decision: RedlineDecision,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct BoundaryDiff {
    /// Mid-tape boundaries (before dispatches 1..n) compared.
    pub compared: usize,
    pub equal: usize,
    pub stronger: usize,
    pub weaker: usize,
    pub redline: Tally,
    pub railgun: Tally,
    pub entry_equal: bool,
    pub exit_equal: bool,
    pub weak: Vec<WeakBoundary>,
    /// Every compared boundary (`class: None` = equal).
    pub rows: Vec<BoundaryRow>,
}

/// One mid-tape boundary on both sides.
#[derive(Clone, Debug, Serialize)]
pub struct BoundaryRow {
    pub dispatch: usize,
    pub previous: String,
    pub symbol: String,
    pub redline: BoundaryState,
    pub railgun: BoundaryState,
    pub class: Option<Class>,
    /// Hazards of railgun's uncovered edges at this boundary.
    pub hazards: Vec<Hazard>,
    pub redline_decision: RedlineDecision,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct IbDiff {
    pub redline_dwords: usize,
    pub railgun_dwords: usize,
    pub railgun_segments: usize,
    /// Identical before canonicalisation.
    pub equal_raw: bool,
    /// Identical after `COMPUTE_USER_DATA_0/1` → dispatch index.
    pub equal_canonical: bool,
    /// Dispatches whose register/dispatch packets differ after
    /// canonicalisation (what runs differs).
    pub state_mismatch: Vec<usize>,
    /// Dispatches whose kernarg buffer address differs.
    pub relocated_user_data: usize,
    /// railgun's own IB boundaries equal its plan, and its NOP pacing its
    /// `post_dispatch_nop` (none before a segment's first dispatch).
    pub plan_matches_ib: bool,
    /// Boundaries (dispatch index; `dispatches` = the exit) whose
    /// post-dispatch NOP dwords differ from Redline's.
    pub pacing_mismatch: Vec<usize>,
    /// Post-dispatch NOP body dwords over the whole tape.
    pub redline_nop_dwords: u64,
    pub railgun_nop_dwords: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CountDiff {
    pub compared: usize,
    pub equal: usize,
    pub different: usize,
    /// Different only in bytes no argument declares (kernargs).
    pub benign_only: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Verdict {
    /// G7 passes only with no weaker difference (no G4 receipt exists) and
    /// railgun's pacing equal to Redline's.
    pub g7_pass: bool,
    pub benign_relocation: usize,
    pub pacing_mismatch: usize,
    pub railgun_stronger: usize,
    pub railgun_weaker: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ShadowDiff {
    pub dispatches: usize,
    pub ib: IbDiff,
    pub kernargs: CountDiff,
    pub bindings: CountDiff,
    pub word_patches: CountDiff,
    pub grid_patches: CountDiff,
    pub boundaries: BoundaryDiff,
    pub differences: Vec<Difference>,
    pub verdict: Verdict,
    /// Errors that stopped a surface from being compared.
    pub errors: Vec<String>,
}

#[derive(Default)]
struct Collector {
    merged: BTreeMap<(Class, Surface, String), Vec<usize>>,
}

impl Collector {
    fn add(&mut self, class: Class, surface: Surface, detail: String, dispatch: Option<usize>) {
        let entry = self.merged.entry((class, surface, detail)).or_default();
        if let Some(d) = dispatch {
            entry.push(d);
        } else if entry.is_empty() {
            entry.push(usize::MAX);
        }
    }

    fn finish(self) -> Vec<Difference> {
        let mut out: Vec<Difference> = self
            .merged
            .into_iter()
            .map(|((class, surface, detail), mut dispatches)| {
                dispatches.retain(|d| *d != usize::MAX);
                Difference { class, surface, detail, count: dispatches.len().max(1), dispatches }
            })
            .collect();
        out.sort_by(|a, b| b.class.cmp(&a.class).then(b.count.cmp(&a.count)));
        out
    }
}

fn describe(b: &BoundaryState) -> String {
    let acquire = match b.acquire() {
        Acquire::None => "no acquire".to_owned(),
        Acquire::Vmem => "vmem acquire".to_owned(),
        Acquire::InterNode => "inter-node acquire".to_owned(),
        Acquire::System => "system acquire".to_owned(),
        Acquire::Other(gcr) => format!("acquire gcr={gcr:#x}"),
    };
    format!("{} + {acquire}", if b.wait_idle { "wait" } else { "no wait" })
}

fn word_kind(role: &WordRole) -> String {
    match role {
        WordRole::GdnFrame { frames } => format!("gdn_frame(frames={frames})"),
    }
}

/// Compare railgun's prepared lowering with Redline's prepared tape.
pub fn diff(program: &KernelProgram, plan: &Pm4Plan, railgun: &RailgunLowering, redline: &RedlineTape) -> ShadowDiff {
    let mut c = Collector::default();
    let mut errors = Vec::new();
    let n = program.nodes.len();
    if redline.dispatches.len() != n {
        errors.push(format!("railgun authored {n} nodes, Redline prepared {} dispatches", redline.dispatches.len()));
    }
    let symbol = |i: usize| program.nodes.get(i).map_or("?", |node| node.symbol.as_str());

    // --- IB -------------------------------------------------------------
    let mut ib = IbDiff {
        redline_dwords: redline.ib.len(),
        railgun_dwords: railgun.segments.iter().map(|s| s.ib.len()).sum(),
        railgun_segments: railgun.segments.len(),
        ..IbDiff::default()
    };
    ib.equal_raw = railgun.segments.len() == 1 && railgun.segments[0].ib == redline.ib;
    ib.redline_nop_dwords = nop_dwords(&redline.ib);
    ib.railgun_nop_dwords = railgun.segments.iter().map(|s| nop_dwords(&s.ib)).sum();
    let redline_parsed = parse(&redline.ib).map_err(|e| errors.push(format!("Redline IB: {e}"))).ok();
    let mut railgun_groups: BTreeMap<usize, Group> = BTreeMap::new();
    let mut railgun_exit: Option<Vec<Packet>> = None;
    for segment in &railgun.segments {
        match parse(&segment.ib) {
            Ok(parsed) if parsed.groups.len() == segment.end - segment.first => {
                for (k, g) in parsed.groups.into_iter().enumerate() {
                    railgun_groups.insert(segment.first + k, g);
                }
                railgun_exit = Some(parsed.epilogue);
            }
            Ok(parsed) => errors.push(format!(
                "railgun segment {}..{}: {} dispatches in its IB",
                segment.first,
                segment.end,
                parsed.groups.len()
            )),
            Err(e) => errors.push(format!("railgun segment {}..{}: {e}", segment.first, segment.end)),
        }
    }
    let mut boundaries = BoundaryDiff::default();
    ib.plan_matches_ib = true;
    let mut canonical_equal = railgun.segments.len() == 1 && errors.is_empty();
    if let Some(parsed) = &redline_parsed {
        if parsed.groups.len() != n {
            errors.push(format!("Redline IB has {} dispatches, the tape {n}", parsed.groups.len()));
            canonical_equal = false;
        }
        for (k, group) in parsed.groups.iter().enumerate().take(n) {
            let (red_b, red_other) = decode_boundary(&group.boundary);
            let (red_state, red_address) = canonical_state(&group.state);
            let Some(rg) = railgun_groups.get(&k) else {
                canonical_equal = false;
                let reason = match &program.nodes[k].lowering {
                    Lowering::HipDirect { reason } | Lowering::Graph { reason } => reason.clone(),
                    Lowering::Pm4 => "no IB group".to_owned(),
                };
                c.add(
                    Class::RailgunStronger,
                    Surface::Ib,
                    format!("{}: lowered outside PM4 with full barriers ({reason})", symbol(k)),
                    Some(k),
                );
                if k > 0 {
                    let railgun_state = BoundaryState::of(plan.boundaries[k].visibility);
                    boundaries.compared += 1;
                    boundaries.stronger += 1;
                    boundaries.redline.add(&red_b);
                    boundaries.railgun.add(&railgun_state);
                    boundaries.rows.push(BoundaryRow {
                        dispatch: k,
                        previous: symbol(k - 1).to_owned(),
                        symbol: symbol(k).to_owned(),
                        redline: BoundaryState { nop_dwords: 0, ..red_b },
                        railgun: railgun_state,
                        class: Some(Class::RailgunStronger),
                        hazards: Vec::new(),
                        redline_decision: redline.dispatches.get(k).map(|d| d.decision.clone()).unwrap_or_default(),
                    });
                }
                continue;
            };
            let (ib_b, rg_other) = decode_boundary(&rg.boundary);
            let planned = &plan.boundaries[k];
            // A segment's first dispatch follows only the entry acquire in its
            // own IB; after a non-PM4 node it also follows the previous
            // segment's exit and the drained HIP stream (the planned barrier).
            let expected = if planned.split { BoundaryState::of(ENTRY) } else { BoundaryState::of(planned.visibility) };
            let expected_nop = if planned.split { 0 } else { railgun.post_dispatch_nop };
            if (BoundaryState { nop_dwords: expected_nop, ..expected }) != ib_b {
                ib.plan_matches_ib = false;
            }
            let rg_b = if planned.split && k > 0 { BoundaryState { nop_dwords: ib_b.nop_dwords, ..BoundaryState::of(planned.visibility) } } else { ib_b };
            let (rg_state, rg_address) = canonical_state(&rg.state);
            if red_state != rg_state {
                canonical_equal = false;
                ib.state_mismatch.push(k);
                c.add(Class::RailgunWeaker, Surface::Ib, format!("{}: register/dispatch packets differ (what runs differs)", symbol(k)), Some(k));
            }
            if red_address != rg_address {
                ib.relocated_user_data += 1;
                c.add(
                    Class::BenignRelocation,
                    Surface::Ib,
                    "COMPUTE_USER_DATA_0/1: each side points at its own kernarg buffer".to_owned(),
                    Some(k),
                );
            }
            let invariant_writes = |g: &Group| {
                g.state.iter().filter(|p| p.opcode == PACKET3_SET_SH_REG && p.dwords.get(1) == Some(&COMPUTE_RESOURCE_LIMITS)).count()
            };
            if invariant_writes(group) != invariant_writes(rg) {
                canonical_equal = false;
                c.add(
                    Class::BenignRelocation,
                    Surface::Ib,
                    "COMPUTE_RESOURCE_LIMITS: a new segment's command buffer re-writes the queue-invariant value".to_owned(),
                    Some(k),
                );
            }
            if red_other != rg_other {
                canonical_equal = false;
                c.add(Class::RailgunWeaker, Surface::Ib, format!("{}: other boundary packets differ ({red_other:x?} vs {rg_other:x?})", symbol(k)), Some(k));
            }
            if group.boundary != rg.boundary {
                canonical_equal = false;
            }
            if k == 0 {
                boundaries.entry_equal = red_b == rg_b;
                if !boundaries.entry_equal {
                    c.add(boundary_class(&red_b, &rg_b), Surface::Boundary, format!("entry: Redline {} vs railgun {}", describe(&red_b), describe(&rg_b)), Some(0));
                }
                continue;
            }
            boundaries.compared += 1;
            boundaries.redline.add(&red_b);
            boundaries.railgun.add(&rg_b);
            let pair = format!("{} -> {}", symbol(k - 1), symbol(k));
            // Pacing follows a PM4 dispatch; a split boundary follows a node
            // lowered outside PM4 (already a counted difference above).
            if red_b.nop_dwords != rg_b.nop_dwords && !planned.split {
                ib.pacing_mismatch.push(k);
                c.add(
                    Class::PacingMismatch,
                    Surface::Ib,
                    format!("post-dispatch NOP dwords before {} (Redline {} vs railgun {})", symbol(k), red_b.nop_dwords, rg_b.nop_dwords),
                    Some(k),
                );
            }
            let red_cmp = BoundaryState { nop_dwords: 0, ..red_b };
            let rg_cmp = BoundaryState { nop_dwords: 0, ..rg_b };
            let decision = redline.dispatches.get(k).map(|d| d.decision.clone()).unwrap_or_default();
            let mut hazards: Vec<_> = plan.boundaries[k].edges.iter().map(|e| e.hazard).collect();
            hazards.sort();
            hazards.dedup();
            let mut row = BoundaryRow {
                dispatch: k,
                previous: symbol(k - 1).to_owned(),
                symbol: symbol(k).to_owned(),
                redline: red_cmp,
                railgun: rg_cmp,
                class: None,
                hazards,
                redline_decision: decision.clone(),
            };
            if red_cmp == rg_cmp {
                boundaries.equal += 1;
                boundaries.rows.push(row);
                continue;
            }
            let class = boundary_class(&red_cmp, &rg_cmp);
            let edges = &plan.boundaries[k].edges;
            let why = if edges.is_empty() {
                "railgun derives no edge".to_owned()
            } else {
                let hazards: BTreeSet<String> = edges.iter().map(|e| format!("{:?}", e.hazard)).collect();
                format!("railgun edges {}", hazards.into_iter().collect::<Vec<_>>().join("+"))
            };
            let redline_why = match (decision.pre_dispatch_name, decision.name_acquire, decision.resource_independent) {
                (true, _, _) => "Redline pre-dispatch name table",
                (false, true, true) => "Redline mid-acquire name table at a resource-independent boundary",
                (false, true, false) => "Redline dependency + mid-acquire name table",
                (false, false, true) => "Redline resource-independent",
                (false, false, false) => "Redline resource dependency",
            };
            c.add(
                class,
                Surface::Boundary,
                format!("{pair}: Redline {} ({redline_why}) vs railgun {} ({why})", describe(&red_cmp), describe(&rg_cmp)),
                Some(k),
            );
            row.class = Some(class);
            boundaries.rows.push(row);
            if class == Class::RailgunWeaker {
                boundaries.weaker += 1;
                boundaries.weak.push(WeakBoundary {
                    dispatch: k,
                    previous: symbol(k - 1).to_owned(),
                    symbol: symbol(k).to_owned(),
                    redline: red_cmp,
                    railgun: rg_cmp,
                    railgun_edges: edges.clone(),
                    redline_decision: decision,
                });
            } else {
                boundaries.stronger += 1;
            }
        }
        let (red_exit, _) = decode_boundary(&parsed.epilogue);
        let rg_exit = railgun_exit.as_ref().map(|p| decode_boundary(p).0);
        boundaries.exit_equal = rg_exit == Some(red_exit) && railgun_exit.as_deref() == Some(parsed.epilogue.as_slice());
        if let Some(rg_exit) = rg_exit {
            if rg_exit.nop_dwords != railgun.post_dispatch_nop {
                ib.plan_matches_ib = false;
            }
        }
        if !boundaries.exit_equal {
            canonical_equal = false;
            if let Some(rg_exit) = rg_exit {
                if rg_exit.nop_dwords != red_exit.nop_dwords {
                    ib.pacing_mismatch.push(n);
                    c.add(
                        Class::PacingMismatch,
                        Surface::Ib,
                        format!("post-dispatch NOP dwords before the exit (Redline {} vs railgun {})", red_exit.nop_dwords, rg_exit.nop_dwords),
                        Some(n),
                    );
                }
                let (red_cmp, rg_cmp) = (BoundaryState { nop_dwords: 0, ..red_exit }, BoundaryState { nop_dwords: 0, ..rg_exit });
                if red_cmp != rg_cmp {
                    c.add(boundary_class(&red_cmp, &rg_cmp), Surface::Boundary, format!("exit: Redline {} vs railgun {}", describe(&red_cmp), describe(&rg_cmp)), None);
                }
            }
        }
    } else {
        canonical_equal = false;
    }
    ib.equal_canonical = canonical_equal;

    // --- kernargs and bindings --------------------------------------------
    let mut kernargs = CountDiff::default();
    let mut bindings = CountDiff::default();
    for (k, red) in redline.dispatches.iter().enumerate().take(n) {
        let node = &program.nodes[k];
        let Some((image, _)) = railgun.kernargs.get(k).and_then(Option::as_ref) else { continue };
        kernargs.compared += 1;
        if *image == red.kernarg {
            kernargs.equal += 1;
        } else {
            kernargs.different += 1;
            if image.len() != red.kernarg.len() {
                c.add(Class::RailgunWeaker, Surface::Kernarg, format!("{}: {} vs {} loader bytes", node.symbol, red.kernarg.len(), image.len()), Some(k));
            } else {
                let mut fields = BTreeSet::new();
                for (at, (a, b)) in red.kernarg.iter().zip(image).enumerate() {
                    if a != b {
                        fields.insert(field_name(program, k, at));
                    }
                }
                kernargs.benign_only += usize::from(fields.iter().all(|(_, declared)| !declared));
                for (field, declared) in fields {
                    // Bytes no explicit or hidden argument declares are never
                    // read by the kernel: what either side leaves there is inert.
                    let class = if declared { Class::RailgunWeaker } else { Class::BenignRelocation };
                    c.add(class, Surface::Kernarg, format!("{}: {field} differs", node.symbol), Some(k));
                }
            }
        }
        let rg_slots: BTreeMap<u32, SlotBinding> = node
            .args
            .iter()
            .filter_map(|(offset, arg)| match arg {
                Arg::Resource { resource, interior } => program.resources.get(resource.0 as usize).map(|b| {
                    (*offset, SlotBinding { offset: *offset, base: b.base, bytes: b.bytes, interior: *interior })
                }),
                _ => None,
            })
            .collect();
        let red_slots: BTreeMap<u32, SlotBinding> = red.bindings.iter().map(|s| (s.offset, *s)).collect();
        for offset in rg_slots.keys().chain(red_slots.keys()).collect::<BTreeSet<_>>() {
            bindings.compared += 1;
            match (red_slots.get(offset), rg_slots.get(offset)) {
                (Some(a), Some(b)) if a == b => bindings.equal += 1,
                (Some(a), Some(b)) => {
                    bindings.different += 1;
                    let wider = b.base <= a.base && b.base + b.bytes >= a.base + a.bytes && b.base + b.interior == a.base + a.interior;
                    let class = if wider { Class::RailgunStronger } else { Class::RailgunWeaker };
                    c.add(class, Surface::Binding, format!("{} +{offset}: Redline binds {:#x}+{} B, railgun {:#x}+{} B", node.symbol, a.base, a.bytes, b.base, b.bytes), Some(k));
                }
                (None, Some(_)) => {
                    bindings.different += 1;
                    c.add(
                        Class::BenignRelocation,
                        Surface::Binding,
                        format!("{} +{offset}: railgun binds a pointer Redline keeps as recorded bytes", node.symbol),
                        Some(k),
                    );
                }
                (Some(_), None) => {
                    bindings.different += 1;
                    c.add(
                        Class::RailgunWeaker,
                        Surface::Binding,
                        format!("{} +{offset}: Redline relocates a pointer railgun keeps constant", node.symbol),
                        Some(k),
                    );
                }
                (None, None) => unreachable!(),
            }
        }
    }

    // --- word and grid patches ---------------------------------------------
    let red_words: BTreeSet<WordPatch> = redline.word_patches.iter().cloned().collect();
    let rg_words: BTreeSet<WordPatch> = program
        .words
        .iter()
        .map(|w| WordPatch { dispatch: w.node, offset: w.offset, kind: word_kind(&w.role) })
        .collect();
    let word_patches = compare_sets(&red_words, &rg_words, &mut c, Surface::WordPatch, |p| (p.dispatch, format!("{} +{} {}", symbol(p.dispatch), p.offset, p.kind)));
    let red_grids: BTreeSet<GridPatch> = redline.grid_patches.iter().cloned().collect();
    // M1 grids are constants: railgun has no grid patches.
    let rg_grids = BTreeSet::new();
    let grid_patches = compare_sets(&red_grids, &rg_grids, &mut c, Surface::GridPatch, |p| {
        (p.dispatch, format!("{} axis {} ceil((pos+{})/{})", symbol(p.dispatch), p.axis, p.addend, p.divisor))
    });

    let differences = c.finish();
    let tally = |class| differences.iter().filter(|d| d.class == class).map(|d| d.count).sum();
    let verdict = Verdict {
        benign_relocation: tally(Class::BenignRelocation),
        pacing_mismatch: tally(Class::PacingMismatch),
        railgun_stronger: tally(Class::RailgunStronger),
        railgun_weaker: tally(Class::RailgunWeaker),
        g7_pass: false,
    };
    let verdict =
        Verdict { g7_pass: verdict.railgun_weaker == 0 && verdict.pacing_mismatch == 0 && errors.is_empty(), ..verdict };
    ShadowDiff { dispatches: n, ib, kernargs, bindings, word_patches, grid_patches, boundaries, differences, verdict, errors }
}

/// Body dwords of every type-3 `NOP` in `ib` (stops at a malformed header;
/// `parse` reports it).
fn nop_dwords(ib: &[u32]) -> u64 {
    let mut total = 0u64;
    let mut cursor = 0usize;
    while let Some(&header) = ib.get(cursor) {
        if header >> 30 != 3 {
            break;
        }
        let body = ((header >> 16) & 0x3fff) as usize + 1;
        if (header >> 8) & 0xff == PACKET3_NOP {
            total += body as u64;
        }
        cursor += 1 + body;
    }
    total
}

fn boundary_class(redline: &BoundaryState, railgun: &BoundaryState) -> Class {
    if railgun.covers(redline) {
        Class::RailgunStronger
    } else {
        Class::RailgunWeaker
    }
}

fn compare_sets<T: Ord>(
    redline: &BTreeSet<T>,
    railgun: &BTreeSet<T>,
    c: &mut Collector,
    surface: Surface,
    label: impl Fn(&T) -> (usize, String),
) -> CountDiff {
    let mut d = CountDiff::default();
    for item in redline.union(railgun) {
        d.compared += 1;
        let (dispatch, text) = label(item);
        match (redline.contains(item), railgun.contains(item)) {
            (true, true) => d.equal += 1,
            (true, false) => {
                d.different += 1;
                c.add(Class::RailgunWeaker, surface, format!("{text}: Redline patches it, railgun does not"), Some(dispatch));
            }
            (false, true) => {
                d.different += 1;
                c.add(Class::RailgunStronger, surface, format!("{text}: railgun patches it, Redline does not"), Some(dispatch));
            }
            (false, false) => unreachable!(),
        }
    }
    d
}

/// The field a kernarg byte belongs to, and whether any argument declares it.
fn field_name(program: &KernelProgram, node: usize, at: usize) -> (String, bool) {
    let n = &program.nodes[node];
    for h in &n.hidden {
        if at >= h.offset as usize && at < (h.offset + h.size) as usize {
            return (format!("{} (+{})", h.value_kind, h.offset), true);
        }
    }
    for (offset, arg) in &n.args {
        let len = match arg {
            Arg::Const(bytes) => bytes.len(),
            Arg::Resource { .. } => 8,
            Arg::Word(_) => 4,
        };
        if at >= *offset as usize && at < *offset as usize + len {
            let kind = match arg {
                Arg::Const(_) => "constant",
                Arg::Resource { .. } => "pointer",
                Arg::Word(_) => "word",
            };
            return (format!("{kind} +{offset}"), true);
        }
    }
    (format!("undeclared byte +{at} (no argument; the kernel never reads it)"), false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet(opcode: u32, body: &[u32]) -> Vec<u32> {
        let mut p = vec![(3 << 30) | ((body.len() as u32 - 1) << 16) | (opcode << 8)];
        p.extend_from_slice(body);
        p
    }

    fn dispatch(address: u64) -> Vec<u32> {
        let mut out = packet(PACKET3_SET_SH_REG, &[COMPUTE_USER_DATA_0, address as u32, (address >> 32) as u32]);
        out.extend(packet(PACKET3_DISPATCH_DIRECT, &[64, 1, 1, 0x8025]));
        out
    }

    fn acquire(gcr: u32) -> Vec<u32> {
        packet(PACKET3_ACQUIRE_MEM, &[0, u32::MAX, 0xff_ffff, 0, 0, 0xa, gcr])
    }

    #[test]
    fn user_data_is_canonicalised_and_boundaries_decode_to_rungs() {
        let mut ib = acquire(GCR_SYSTEM_GFX12);
        ib.extend(dispatch(0x1000));
        ib.extend(packet(PACKET3_EVENT_WRITE, &[CS_PARTIAL_FLUSH]));
        ib.extend(acquire(GCR_INTER_NODE));
        ib.extend(dispatch(0x2000));
        let parsed = parse(&ib).unwrap();
        assert_eq!(parsed.groups.len(), 2);
        let (b, _) = decode_boundary(&parsed.groups[1].boundary);
        assert_eq!(b, BoundaryState { wait_idle: true, acquire: Some(Acquire::InterNode), nop_dwords: 0 });
        let (state, address) = canonical_state(&parsed.groups[1].state);
        assert_eq!(address, Some(0x2000));
        assert_eq!(state, canonical_state(&parse(&dispatch(0x9999)).unwrap().groups[0].state).0);
    }

    #[test]
    fn weaker_means_less_wait_or_an_incomparable_or_lower_rung() {
        let s = |wait, acquire| BoundaryState { wait_idle: wait, acquire, nop_dwords: 0 };
        let redline = s(true, Some(Acquire::InterNode));
        assert_eq!(boundary_class(&redline, &s(true, Some(Acquire::System))), Class::RailgunStronger);
        assert_eq!(boundary_class(&redline, &s(false, Some(Acquire::InterNode))), Class::RailgunWeaker);
        assert_eq!(boundary_class(&redline, &s(true, Some(Acquire::Vmem))), Class::RailgunWeaker);
        assert_eq!(boundary_class(&s(false, Some(Acquire::InterNode)), &s(false, None)), Class::RailgunWeaker);
    }

    /// Two independent nodes: railgun elides the boundary Redline waits at,
    /// and Redline also patches a word railgun keeps constant. Both are
    /// weaker; the kernarg-buffer addresses are benign; G7 fails.
    #[test]
    fn elided_wait_and_missing_patch_fail_g7_while_relocation_is_benign() {
        use crate::kernel::tests::{launch, ptr};
        use crate::kernel::{author, MemClass::VmemLoad};
        use crate::plan::{plan, CacheTable, RowPolicy};
        let launches = vec![
            launch(&[(0, 0x7000_0000)], 16, vec![ptr(0, &[VmemLoad])]),
            launch(&[(0, 0x7100_0000)], 16, vec![ptr(0, &[VmemLoad])]),
        ];
        let program = author("gfx1201", &launches);
        let p = plan(&program, &CacheTable::for_arch("gfx1201").unwrap(), RowPolicy::TierD);
        assert!(p.boundaries[1].visibility.is_empty());
        let exit = |ib: &mut Vec<u32>| {
            ib.extend(packet(PACKET3_EVENT_WRITE, &[CS_PARTIAL_FLUSH]));
            ib.extend(acquire(GCR_SYSTEM_GFX12));
        };
        let mut railgun_ib = acquire(GCR_SYSTEM_GFX12);
        railgun_ib.extend(dispatch(0x1000));
        railgun_ib.extend(dispatch(0x1100));
        exit(&mut railgun_ib);
        let mut redline_ib = acquire(GCR_SYSTEM_GFX12);
        redline_ib.extend(dispatch(0x2000));
        redline_ib.extend(packet(PACKET3_EVENT_WRITE, &[CS_PARTIAL_FLUSH]));
        redline_ib.extend(acquire(GCR_INTER_NODE));
        redline_ib.extend(dispatch(0x2100));
        exit(&mut redline_ib);
        let images: Vec<_> = (0..2).map(|i| program.kernarg_image(i, 16, &[]).unwrap()).collect();
        let railgun = RailgunLowering {
            segments: vec![SegmentIb { first: 0, end: 2, ib: railgun_ib }],
            kernargs: images.iter().zip([0x1000, 0x1100]).map(|(k, a)| Some((k.clone(), a))).collect(),
            post_dispatch_nop: 0,
        };
        let dispatch_of = |i: usize, address| RedlineDispatch {
            symbol: "k".into(),
            kernarg: images[i].clone(),
            kernarg_address: address,
            bindings: Vec::new(),
            decision: RedlineDecision::default(),
        };
        let redline = RedlineTape {
            ib: redline_ib,
            dispatches: vec![dispatch_of(0, 0x2000), dispatch_of(1, 0x2100)],
            word_patches: vec![WordPatch { dispatch: 1, offset: 8, kind: "position+1".into() }],
            grid_patches: Vec::new(),
        };
        let d = diff(&program, &p, &railgun, &redline);
        assert!(d.errors.is_empty(), "{:?}", d.errors);
        assert!(!d.verdict.g7_pass);
        assert_eq!((d.boundaries.equal, d.boundaries.weaker, d.boundaries.weak[0].dispatch), (0, 1, 1));
        assert_eq!(d.word_patches.different, 1);
        assert_eq!(d.ib.relocated_user_data, 2);
        assert!(d.ib.state_mismatch.is_empty() && d.boundaries.entry_equal && d.boundaries.exit_equal);
        assert_eq!(d.verdict.benign_relocation, 2 + 2, "two relocated kernarg addresses, two railgun-only pointer bindings");
        assert_eq!(d.verdict.railgun_weaker, 2);
    }

    /// Railgun's pacing must equal Redline's: the same tape with and
    /// without the post-dispatch NOPs differs only in pacing, and that alone
    /// fails G7; paced like Redline, the tapes are canonically equal.
    #[test]
    fn post_dispatch_nop_pacing_must_equal_redline() {
        use crate::kernel::tests::{launch, ptr};
        use crate::kernel::{author, MemClass::VmemLoad};
        use crate::plan::{plan, ArchLowering, CacheTable, RowPolicy};
        let launches = vec![
            launch(&[(0, 0x7000_0000)], 16, vec![ptr(0, &[VmemLoad])]),
            launch(&[(0, 0x7100_0000)], 16, vec![ptr(0, &[VmemLoad])]),
        ];
        let program = author("gfx1201", &launches);
        let p = plan(&program, &CacheTable::for_arch("gfx1201").unwrap(), RowPolicy::TierD);
        let nop = ArchLowering::for_arch("gfx1201").post_dispatch_nop;
        assert_eq!(nop, 64);
        assert_eq!(ArchLowering::for_arch("gfx1100").post_dispatch_nop, 0);
        let tape = |pace: u32| {
            let mut ib = acquire(GCR_SYSTEM_GFX12);
            for address in [0x1000, 0x1100] {
                ib.extend(dispatch(address));
                if pace > 0 {
                    ib.extend(packet(PACKET3_NOP, &vec![0; pace as usize]));
                }
            }
            ib.extend(packet(PACKET3_EVENT_WRITE, &[CS_PARTIAL_FLUSH]));
            ib.extend(acquire(GCR_SYSTEM_GFX12));
            ib
        };
        let images: Vec<_> = (0..2).map(|i| program.kernarg_image(i, 16, &[]).unwrap()).collect();
        let redline = RedlineTape {
            ib: tape(nop),
            dispatches: (0..2)
                .map(|i| RedlineDispatch {
                    symbol: "k".into(),
                    kernarg: images[i].clone(),
                    kernarg_address: [0x1000, 0x1100][i],
                    bindings: Vec::new(),
                    decision: RedlineDecision::default(),
                })
                .collect(),
            word_patches: Vec::new(),
            grid_patches: Vec::new(),
        };
        let lowering = |pace: u32| RailgunLowering {
            segments: vec![SegmentIb { first: 0, end: 2, ib: tape(pace) }],
            kernargs: images.iter().zip([0x1000, 0x1100]).map(|(k, a)| Some((k.clone(), a))).collect(),
            post_dispatch_nop: pace,
        };
        let paced = diff(&program, &p, &lowering(nop), &redline);
        assert!(paced.errors.is_empty(), "{:?}", paced.errors);
        assert!(paced.verdict.g7_pass && paced.ib.equal_raw && paced.ib.plan_matches_ib);
        assert_eq!((paced.ib.redline_nop_dwords, paced.ib.railgun_nop_dwords), (128, 128));
        let unpaced = diff(&program, &p, &lowering(0), &redline);
        assert!(!unpaced.verdict.g7_pass && unpaced.ib.plan_matches_ib);
        assert_eq!(unpaced.ib.pacing_mismatch, vec![1, 2], "before dispatch 1 and before the exit");
        assert_eq!((unpaced.verdict.pacing_mismatch, unpaced.verdict.railgun_weaker), (2, 0));
    }
}
