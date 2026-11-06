// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Admission of a program: `Certified`, or the one lab-only state
//! [`LAB_PENDING`] — "differential-verified, certificate pending".
//!
//! A persistent megakernel's task-graph certificate may not exist yet. The lab
//! state is **not a bypass**: the node goes through the same pipeline (author
//! → plan → lower → submit), with the certificate slot marked pending instead
//! of proven, and it flips to proven when the certifier lands (a receipt
//! carrying the task-graph facts makes [`crate::kernel::author`] yield `Pm4`
//! directly; nothing here changes). Until then:
//!
//! - the state exists only under `--features lab`: the constructors
//!   ([`KernelProgram::admit_lab_pending`], `GroupOrdering::lab_pending`) are
//!   `cfg(feature = "lab")`, and every product consumer
//!   ([`crate::kernel::Lowering::is_pm4`], so `WindowPlan::new`, the shadow
//!   diff and the mq6 route) refuses a `LabPending` node unchanged;
//! - it is never silent: [`KernelProgram::admission`], [`Pm4Plan::admission`]
//!   and [`GroupPlan::admission`] report it, and [`Admission::receipt`] — the
//!   text the harness must put in **every** receipt — refuses to render a
//!   `LabPending` admission without a [`DifferentialWitness`], the
//!   byte-identity of the node's output against Eager on the same fixture.
//!
//! # Completeness of GPU-generated task spans
//!
//! A task-graph proof with span lists ([`crate::task_graph`], span mode)
//! proves **safety** only: no out-of-grid or duplicate tile runs, the count
//! never overflows, the GPU writes no authority. It does not prove that the
//! producer's live list holds every live tile; that is a functional property
//! of EXPAND_LIVE. So the admission of such a program is
//! [`Admission::Spans`] (over whatever its certificates say), and it renders
//! a receipt only through [`Admission::receipt_with_completeness`] with the
//! harness-attached [`CompletenessEvidence`] (producer, byte oracle, P0 vs
//! eager), as `"completeness": ` [`COMPLETENESS_FUNCTIONAL`]; without it
//! (or via [`Admission::receipt`]) the receipt refuses.
//!
//! # What `admit_lab_pending` accepts
//!
//! The node must be a registered, receipted kernel (it has a `kref`) whose
//! lowering is `HipDirect` or `Conditional` **only** for certificate reasons.
//! Every reason must be one of [`LAB_PENDING_ALLOWED`]; any other (a missing
//! receipt, a kernarg mismatch, an unprobed pointer, a failing G8 or guard,
//! hidden arguments railgun does not supply, an unconsumed declared word, a
//! non-allocation sealed table...) is a refusal that no lab state may
//! override.

use std::ops::Range;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::coresident::CoResidentGroup;
use crate::kernel::{KernelProgram, Lowering};
use crate::plan::Pm4Plan;

/// The admission state a lab-pending program reports and logs.
pub const LAB_PENDING: &str = "lab: differential-verified, certificate pending";

/// The only lowering reasons (`"; "`-separated fragments of a `HipDirect` or
/// `Conditional` reason) that [`KernelProgram::admit_lab_pending`] treats as
/// "a certificate has not been produced yet". Each entry is
/// `(fragment, how it matches)`: an `A1 Unknown (...)` prefix, or an exact
/// string.
pub const LAB_PENDING_ALLOWED: &[(&str, Match)] = &[
    // The A1 analysis could not decide the kernel's memory effects (a
    // persistent kernel loading pointers from an arena): the certificate is
    // what would decide them.
    ("A1 Unknown (", Match::PrefixParen),
    // The descriptor-arena records are not bound (`bind_records`).
    (crate::kernel::ARENA_UNBOUND, Match::Exact),
    // A `Conditional` kernel's bind witnesses have not been consumed.
    ("indexed_bind witness not consumed", Match::Exact),
    ("sealed_table witness not consumed", Match::Exact),
    (
        "indexed_bind and sealed_table witnesses not consumed",
        Match::Exact,
    ),
];

/// How a [`LAB_PENDING_ALLOWED`] entry matches a reason fragment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Match {
    /// The fragment equals the entry.
    Exact,
    /// The fragment starts with the entry and ends with the closing `)`.
    PrefixParen,
}

/// Whether one `"; "`-separated reason fragment is a pending certificate.
pub fn is_pending_certificate_reason(fragment: &str) -> bool {
    LAB_PENDING_ALLOWED.iter().any(|(entry, how)| match how {
        Match::Exact => fragment == *entry,
        Match::PrefixParen => fragment.starts_with(entry) && fragment.ends_with(')'),
    })
}

/// What the receipt says about a program with GPU-generated task spans: the
/// certificate proves **safety** (no out-of-grid or duplicate tile, no count
/// overflow, no GPU-written authority), never that the producer's live list
/// holds every live tile.
pub const COMPLETENESS_FUNCTIONAL: &str = "functional (EXPAND_LIVE), not certified";

/// Whether a program (or plan, or group) is certified, or lab-pending.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum Admission {
    /// Every node and group ordering rests on a certificate.
    Certified,
    /// At least one node or group ordering awaits its certificate; `reasons`
    /// names each (the exact refusals the certificate would discharge).
    LabPending { reasons: Vec<String> },
    /// `base`, for a program with `lists` GPU-generated task span lists
    /// ([`crate::task_graph::TaskGraphProof::spans`]). Whatever `base` says,
    /// the receipt also names the completeness of the live lists, which no
    /// certificate covers: it needs harness evidence
    /// ([`CompletenessEvidence`], [`Admission::receipt_with_completeness`]).
    Spans { base: Box<Admission>, lists: usize },
}

impl Admission {
    /// `Certified` for no pending reason, else `LabPending`.
    pub fn from_reasons(reasons: Vec<String>) -> Self {
        if reasons.is_empty() {
            Self::Certified
        } else {
            Self::LabPending { reasons }
        }
    }

    /// This admission for a program with `lists` span lists (itself for
    /// none).
    pub fn with_spans(self, lists: usize) -> Self {
        match self {
            _ if lists == 0 => self,
            Self::Spans { base, lists: have } => Self::Spans {
                base,
                lists: have + lists,
            },
            base => Self::Spans {
                base: Box::new(base),
                lists,
            },
        }
    }

    /// The number of span lists whose completeness the receipt must name.
    pub fn spans(&self) -> usize {
        match self {
            Self::Spans { lists, .. } => *lists,
            _ => 0,
        }
    }

    /// Safety rests on certificates only. (Completeness of span lists never
    /// does: [`Admission::spans`].)
    pub fn is_certified(&self) -> bool {
        match self {
            Self::Certified => true,
            Self::LabPending { .. } => false,
            Self::Spans { base, .. } => base.is_certified(),
        }
    }

    /// The receipt record the harness logs. `Certified` is
    /// `{"admission":"proven"}`. `LabPending` needs the eager byte-identity
    /// `witness` ("lab-pending results need the eager byte-identity witness"
    /// otherwise) and renders [`LAB_PENDING`], `"certified": false`, the
    /// pending reasons and the differential digests. An admission with span
    /// lists never renders here: it needs
    /// [`Admission::receipt_with_completeness`].
    pub fn receipt(
        &self,
        witness: Option<&DifferentialWitness>,
    ) -> Result<AdmissionReceipt, String> {
        self.receipt_with_completeness(witness, None)
    }

    /// [`Admission::receipt`] with the harness-attached completeness
    /// `evidence`. A program with span lists renders only with it
    /// (`"completeness": ` [`COMPLETENESS_FUNCTIONAL`] and the evidence next
    /// to the base admission); evidence for a program without spans is
    /// refused, so a receipt never claims evidence it has nothing to say
    /// about.
    pub fn receipt_with_completeness(
        &self,
        witness: Option<&DifferentialWitness>,
        evidence: Option<&CompletenessEvidence>,
    ) -> Result<AdmissionReceipt, String> {
        match self {
            Self::Spans { base, lists } => {
                let evidence = evidence.ok_or_else(|| {
                    format!(
                        "{lists} span list(s): completeness is {COMPLETENESS_FUNCTIONAL} and needs the producer, oracle and P0-vs-eager evidence"
                    )
                })?;
                evidence.validate()?;
                let mut receipt = base.receipt_with_completeness(witness, None)?;
                receipt.completeness = Some(COMPLETENESS_FUNCTIONAL);
                receipt.completeness_evidence = Some(evidence.clone());
                Ok(receipt)
            }
            _ if evidence.is_some() => {
                Err("completeness evidence given for a program without GPU-generated spans".into())
            }
            Self::Certified => Ok(AdmissionReceipt {
                admission: "proven",
                certified: None,
                pending: None,
                differential: None,
                completeness: None,
                completeness_evidence: None,
            }),
            Self::LabPending { reasons } => {
                let witness = witness.ok_or_else(|| {
                    "lab-pending results need the eager byte-identity witness".to_owned()
                })?;
                Ok(AdmissionReceipt {
                    admission: LAB_PENDING,
                    certified: Some(false),
                    pending: Some(reasons.clone()),
                    differential: Some(witness.clone()),
                    completeness: None,
                    completeness_evidence: None,
                })
            }
        }
    }
}

/// What [`Admission::receipt`] renders (serializes to the JSON the harness
/// writes into every receipt).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AdmissionReceipt {
    /// `"proven"` or [`LAB_PENDING`].
    pub admission: &'static str,
    /// `Some(false)` for a lab-pending admission; absent when proven.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub certified: Option<bool>,
    /// The pending certificate reasons; absent when proven.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pending: Option<Vec<String>>,
    /// The eager byte-identity witness; absent when proven.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub differential: Option<DifferentialWitness>,
    /// [`COMPLETENESS_FUNCTIONAL`] for a program with span lists; absent
    /// otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completeness: Option<&'static str>,
    /// The evidence behind `completeness`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completeness_evidence: Option<CompletenessEvidence>,
}

/// The harness-attached evidence that a span producer's live list is the
/// eager live set (the one thing the certificate does not prove). Railgun
/// cannot check it; it only refuses to render without it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CompletenessEvidence {
    /// The producer (EXPAND_LIVE) the evidence is about: template, object
    /// digest or job id.
    pub producer: String,
    /// The byte oracle that checked the live list (job id or digest).
    pub oracle: String,
    /// The P0 (railgun, spans) vs eager byte-identity run (job id or digest).
    pub p0_vs_eager: String,
}

impl CompletenessEvidence {
    /// Every field names something.
    fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("producer", &self.producer),
            ("oracle", &self.oracle),
            ("p0_vs_eager", &self.p0_vs_eager),
        ] {
            if value.trim().is_empty() {
                return Err(format!("completeness evidence {name} is empty"));
            }
        }
        Ok(())
    }
}

/// The railgun output was compared with Eager's on the same fixture and the
/// two are byte-identical. Only [`DifferentialWitness::compare`] builds one.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DifferentialWitness {
    fixture_sha256: String,
    eager_sha256: String,
    railgun_sha256: String,
}

impl DifferentialWitness {
    /// Compare the Eager output with the railgun output of the same
    /// `fixture` (its input bytes, digested to identify it). `Err` unless the
    /// outputs are byte-identical; empty outputs prove nothing and are
    /// refused.
    pub fn compare(
        fixture: &[u8],
        eager_output: &[u8],
        railgun_output: &[u8],
    ) -> Result<Self, String> {
        if eager_output.is_empty() || railgun_output.is_empty() {
            return Err("differential witness over an empty output proves nothing".to_owned());
        }
        if eager_output != railgun_output {
            let at = eager_output
                .iter()
                .zip(railgun_output)
                .position(|(a, b)| a != b);
            return Err(match at {
                Some(at) => format!(
                    "railgun output differs from eager at byte {at} ({} vs {} bytes)",
                    eager_output.len(),
                    railgun_output.len()
                ),
                None => format!(
                    "railgun output is {} bytes, eager {} (common prefix identical)",
                    railgun_output.len(),
                    eager_output.len()
                ),
            });
        }
        Ok(Self {
            fixture_sha256: hex(fixture),
            eager_sha256: hex(eager_output),
            railgun_sha256: hex(railgun_output),
        })
    }

    pub fn fixture_sha256(&self) -> &str {
        &self.fixture_sha256
    }

    pub fn eager_sha256(&self) -> &str {
        &self.eager_sha256
    }

    pub fn railgun_sha256(&self) -> &str {
        &self.railgun_sha256
    }
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `node {i} ({symbol}): {pending}` for each `LabPending` node of `range`.
pub(crate) fn node_pending(program: &KernelProgram, range: Range<usize>) -> Vec<String> {
    let mut out = Vec::new();
    for i in range {
        if let Lowering::LabPending { pending } = &program.nodes[i].lowering {
            out.push(format!("node {i} ({}): {pending}", program.nodes[i].symbol));
        }
    }
    out
}

/// The reasons a group's ordering is uncertified (empty when certified): the
/// generic one, then each named reason the ordering carries.
pub(crate) fn ordering_pending(index: usize, g: &CoResidentGroup) -> Vec<String> {
    if !g.ordering.is_pending() {
        return Vec::new();
    }
    let mut out = vec![format!(
        "group {index} {:?}: ordering edges are not from a task-graph certificate",
        g.nodes
    )];
    out.extend(
        g.ordering
            .pending_reasons()
            .iter()
            .map(|r| format!("group {index} {:?}: {r}", g.nodes)),
    );
    out
}

/// Every pending reason of `program`: its `LabPending` nodes, then its
/// uncertified group orderings.
pub(crate) fn program_pending(program: &KernelProgram) -> Vec<String> {
    let mut out = node_pending(program, 0..program.nodes.len());
    out.extend(
        program
            .groups
            .iter()
            .enumerate()
            .flat_map(|(i, g)| ordering_pending(i, g)),
    );
    out
}

impl KernelProgram {
    /// [`Admission::Certified`], or [`Admission::LabPending`] naming every
    /// `LabPending` node and uncertified group ordering; with span lists, an
    /// [`Admission::Spans`] over it.
    pub fn admission(&self) -> Admission {
        Admission::from_reasons(program_pending(self)).with_spans(
            self.groups
                .iter()
                .map(|g| g.ordering.spans())
                .sum::<usize>(),
        )
    }
}

impl Pm4Plan {
    /// The admission of the program this plan was built from (recorded by
    /// [`crate::plan::plan`]).
    pub fn admission(&self) -> Admission {
        Admission::from_reasons(self.pending.clone())
            .with_spans(self.groups.iter().map(|g| g.spans).sum::<usize>())
    }
}

impl crate::coresident::GroupPlan {
    /// The admission of this group: its `LabPending` members and its ordering.
    pub fn admission(&self) -> Admission {
        Admission::from_reasons(self.pending.clone()).with_spans(self.spans)
    }
}

#[cfg(feature = "lab")]
mod lab {
    use std::collections::BTreeSet;

    use crate::kernel::{Arg, KernelProgram, Lowering, MemClass, RegionEffect};
    use crate::{ResourceId, ResourceKind};

    use super::is_pending_certificate_reason;

    /// Classes granted on an event-cell resource reached through arena
    /// records: the event contract's whole vocabulary (never a plain store).
    const EVENT_CLASSES: [MemClass; 4] = [
        MemClass::VmemLoad,
        MemClass::LoadAcquire,
        MemClass::AtomicRelease,
        MemClass::AtomicCommutative,
    ];

    fn grant(
        effects: &mut Vec<RegionEffect>,
        arg_offset: u32,
        resource: ResourceId,
        add: &[MemClass],
    ) {
        match effects
            .iter_mut()
            .find(|e| e.arg_offset == arg_offset && e.resource == resource)
        {
            Some(e) => {
                for c in add {
                    if !e.classes.contains(c) {
                        e.classes.push(*c);
                    }
                }
            }
            None => effects.push(RegionEffect {
                arg_offset,
                resource,
                classes: add.to_vec(),
            }),
        }
    }

    impl KernelProgram {
        /// Admit node `node` in the lab state
        /// [`super::LAB_PENDING`]: it stays out of every product route
        /// (`Lowering::LabPending` is not `is_pm4`) and is PM4-lowerable only
        /// through `Lowering::is_pm4_lowerable`.
        ///
        /// Refused (the program is unchanged) unless the node has a `kref`, is
        /// `HipDirect` or `Conditional`, and **every** `"; "`-separated
        /// fragment of its reason is in [`super::LAB_PENDING_ALLOWED`] —
        /// `A1 Unknown (...)`, `descriptor_arena records not bound`, or an
        /// unconsumed indexed_bind/sealed_table witness. A `Graph`/`Pm4`
        /// node, a node with no receipt, or any other refusal (kernarg
        /// mismatch, unprobed pointer, failing G8 or guard, hidden
        /// arguments, declared words) is refused with the offending reason.
        ///
        /// Effects become conservative; A1's per-argument effects are kept
        /// and added to:
        /// - every explicit pointer resource that is not declared
        ///   `EventCells` gains `VmemLoad` + `VmemStore` — except a declared
        ///   `BindingRecords` arena, which gains `VmemLoad` only (records
        ///   are immutable during replay; a node whose A1 effects write one
        ///   is refused, as in `bind_records`). `EventCells` keep A1's
        ///   classes and are still validated against the event contract;
        /// - if `arena` is given it must be declared `BindingRecords` and
        ///   some argument of the node must be bound to it; every resource a
        ///   record field names then gains `VmemLoad` + `VmemStore` with that
        ///   argument's offset — except `EventCells`, which gain the event
        ///   classes (`VmemLoad`, `LoadAcquire`, `AtomicRelease`,
        ///   `AtomicCommutative`; a plain store would break their contract).
        pub fn admit_lab_pending(
            &mut self,
            node: usize,
            arena: Option<ResourceId>,
        ) -> Result<(), String> {
            let n = self
                .nodes
                .get(node)
                .ok_or_else(|| format!("lab-pending: no node {node}"))?;
            let who = format!("lab-pending: node {node} ({})", n.symbol);
            let reason = match &n.lowering {
                Lowering::HipDirect { reason } | Lowering::Conditional { reason } => reason,
                Lowering::Pm4 => return Err(format!("{who} is already PM4: nothing is pending")),
                Lowering::Graph { reason } => {
                    return Err(format!(
                        "{who} is certified but not PM4 ({reason}): not a pending certificate"
                    ))
                }
                Lowering::LabPending { .. } => return Err(format!("{who} is already lab-pending")),
            };
            if n.kref.is_none() {
                return Err(format!(
                    "{who} has no registered receipted object (receipt_missing): {reason}"
                ));
            }
            if reason.is_empty() {
                return Err(format!("{who} has no refusal reason: nothing is pending"));
            }
            if let Some(bad) = reason
                .split("; ")
                .find(|fragment| !is_pending_certificate_reason(fragment))
            {
                return Err(format!(
                    "{who}: {bad:?} is not a pending certificate (allow-list: A1 Unknown, \
                     descriptor_arena records not bound, indexed_bind/sealed_table witness \
                     not consumed); full reason: {reason}"
                ));
            }

            let mut effects = n.effects.clone();
            let kind = |r: ResourceId| self.kinds.get(&r);
            // Explicit pointer arguments, in offset order.
            for (offset, arg) in &n.args {
                let Arg::Resource { resource, .. } = arg else {
                    continue;
                };
                match kind(*resource) {
                    Some(ResourceKind::EventCells { .. }) => {}
                    Some(ResourceKind::BindingRecords(_)) => {
                        if effects
                            .iter()
                            .any(|e| e.resource == *resource && e.writes())
                        {
                            return Err(format!(
                                "{who} writes binding-record arena {}: records are immutable \
                                 during replay",
                                resource.0
                            ));
                        }
                        grant(&mut effects, *offset, *resource, &[MemClass::VmemLoad]);
                    }
                    _ => grant(
                        &mut effects,
                        *offset,
                        *resource,
                        &[MemClass::VmemLoad, MemClass::VmemStore],
                    ),
                }
            }
            if let Some(arena) = arena {
                let Some(ResourceKind::BindingRecords(records)) = self.kinds.get(&arena) else {
                    return Err(format!(
                        "{who}: arena resource {} is not declared BindingRecords",
                        arena.0
                    ));
                };
                let Some(arena_arg) = n.args.iter().find_map(|(offset, arg)| {
                    matches!(arg, Arg::Resource { resource, .. } if *resource == arena)
                        .then_some(*offset)
                }) else {
                    return Err(format!(
                        "{who}: no argument of the node is bound to arena {}",
                        arena.0
                    ));
                };
                let fields: BTreeSet<ResourceId> = records
                    .iter()
                    .flat_map(|r| r.fields.iter().map(|f| f.resource))
                    .collect();
                for resource in fields {
                    match kind(resource) {
                        Some(ResourceKind::EventCells { .. }) => {
                            grant(&mut effects, arena_arg, resource, &EVENT_CLASSES)
                        }
                        _ => grant(
                            &mut effects,
                            arena_arg,
                            resource,
                            &[MemClass::VmemLoad, MemClass::VmemStore],
                        ),
                    }
                }
            }
            // Event cells keep the event contract whatever was granted.
            for e in &effects {
                if !matches!(kind(e.resource), Some(ResourceKind::EventCells { .. })) {
                    continue;
                }
                if let Some(class) = e.classes.iter().find(|c| !c.is_event_class()) {
                    return Err(format!(
                        "{who} accesses event cells (resource {}) with {class:?}: not an event \
                         class",
                        e.resource.0
                    ));
                }
            }

            let pending = reason.clone();
            let n = &mut self.nodes[node];
            n.effects = effects;
            n.lowering = Lowering::LabPending { pending };
            Ok(())
        }
    }
}

#[cfg(all(test, feature = "lab"))]
mod lab_tests {
    use super::*;
    use crate::coresident::{GroupOrdering, OrderedEdge, WHY_ORDERED_EDGE};
    use crate::kernel::tests::{facts, launch, ptr};
    use crate::kernel::{author, FactStatus, MemClass::*};
    use crate::plan::{plan, CacheTable, RowPolicy};
    use crate::sealed_table::SealedTableRegistry;
    use crate::{BindingField, BindingRecord, ResourceId, ResourceKind};

    const A: u64 = 0x7000_0000;
    const E: u64 = 0x7100_0000;
    const D: u64 = 0x7200_0000;

    fn plan_of(p: &KernelProgram) -> Pm4Plan {
        plan(
            p,
            &CacheTable::for_arch("gfx1201").unwrap(),
            RowPolicy::TierD,
        )
    }

    /// A launch whose facts are `A1 Unknown`.
    fn unknown(
        pointers: &[(u32, u64)],
        explicit: u32,
        args: Vec<crate::kernel::ArgFact>,
    ) -> crate::kernel::LaunchRecord {
        let mut l = launch(pointers, explicit, args.clone());
        let mut f = (*facts(args, explicit)).clone();
        f.facts.status = FactStatus::Unknown(vec!["indirect-pointer-load".into()]);
        l.facts = Ok(std::sync::Arc::new(f));
        l
    }

    #[test]
    fn a1_unknown_node_is_admitted_lab_pending_with_conservative_effects() {
        let mut p = author(
            "gfx1201",
            &[unknown(
                &[(0, A), (8, E)],
                16,
                vec![ptr(0, &[SmemLoad]), ptr(8, &[AtomicRelease])],
            )],
        );
        p.declare_event_cells(ResourceId(1), 4).unwrap();
        assert!(matches!(
            &p.nodes[0].lowering,
            Lowering::HipDirect { reason } if reason.starts_with("A1 Unknown (")
        ));
        assert_eq!(p.admission(), Admission::Certified, "no lab node yet");
        p.admit_lab_pending(0, None).unwrap();
        let n = &p.nodes[0];
        assert!(
            matches!(&n.lowering, Lowering::LabPending { pending } if pending.starts_with("A1 Unknown ("))
        );
        assert!(!n.lowering.is_pm4());
        assert!(n.lowering.is_pm4_lowerable());
        let on = |r: u32| {
            n.effects
                .iter()
                .find(|e| e.resource == ResourceId(r))
                .unwrap()
        };
        // A's A1 class is kept and VmemLoad + VmemStore are added.
        assert!(on(0).has(SmemLoad) && on(0).has(VmemLoad) && on(0).has(VmemStore));
        // The event cells keep A1's classes: no plain store.
        assert_eq!(on(1).classes, vec![AtomicRelease]);
        match p.admission() {
            Admission::LabPending { reasons } => {
                assert_eq!(reasons.len(), 1);
                assert!(
                    reasons[0].starts_with("node 0 (k): A1 Unknown ("),
                    "{reasons:?}"
                );
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(plan_of(&p).admission(), p.admission());
    }

    #[test]
    fn a_plain_store_on_event_cells_is_still_refused() {
        let mut p = author(
            "gfx1201",
            &[unknown(&[(0, E)], 8, vec![ptr(0, &[VmemStore])])],
        );
        let before = p.clone();
        // Declaring as event cells already fails; bypass it to hit admit's own check.
        p.kinds
            .insert(ResourceId(0), ResourceKind::EventCells { cells: 4 });
        let err = p.admit_lab_pending(0, None).unwrap_err();
        assert!(err.contains("not an event class"), "{err}");
        assert_eq!(p.nodes, before.nodes, "an Err leaves the nodes unchanged");
    }

    #[test]
    fn only_certificate_reasons_are_pending() {
        // receipt_missing: no kref.
        let mut missing = launch(&[(0, A)], 8, vec![ptr(0, &[VmemLoad])]);
        missing.facts = Err("no receipt for k".into());
        let mut p = author("gfx1201", &[missing]);
        let err = p.admit_lab_pending(0, None).unwrap_err();
        assert!(err.contains("receipt_missing"), "{err}");
        assert!(matches!(p.nodes[0].lowering, Lowering::HipDirect { .. }));

        // Unknown AND an unprobed pointer: the pointer refusal is not pending.
        let mut l = unknown(&[(0, A)], 8, vec![ptr(0, &[VmemLoad])]);
        l.allocations.clear();
        let mut p = author("gfx1201", &[l]);
        let err = p.admit_lab_pending(0, None).unwrap_err();
        assert!(err.contains("was not probed"), "{err}");

        // Unknown AND a failing G8 differential.
        let mut l = unknown(&[(0, A)], 8, vec![ptr(0, &[VmemLoad])]);
        let mut f = (*l.facts.clone().unwrap()).clone();
        f.facts.differential = "fail".into();
        l.facts = Ok(std::sync::Arc::new(f));
        let mut p = author("gfx1201", &[l]);
        let err = p.admit_lab_pending(0, None).unwrap_err();
        assert!(err.contains("G8 differential fails"), "{err}");

        // A proven PM4 node and a missing node.
        let mut p = author(
            "gfx1201",
            &[launch(&[(0, A)], 8, vec![ptr(0, &[VmemLoad])])],
        );
        assert!(p
            .admit_lab_pending(0, None)
            .unwrap_err()
            .contains("already PM4"));
        assert!(p.admit_lab_pending(3, None).is_err());

        // Twice.
        let mut p = author(
            "gfx1201",
            &[unknown(&[(0, A)], 8, vec![ptr(0, &[VmemLoad])])],
        );
        p.admit_lab_pending(0, None).unwrap();
        assert!(p
            .admit_lab_pending(0, None)
            .unwrap_err()
            .contains("already lab-pending"));
    }

    #[test]
    fn arena_fields_get_conservative_effects_and_the_arena_stays_read_only() {
        // n0 (Unknown) reads arena resource 0 through arg 0; n1 touches X (1), Y (2), E (3).
        let n0 = unknown(&[(0, A)], 8, vec![ptr(0, &[SmemLoad])]);
        let toucher = launch(
            &[(0, 0x7300_0000), (8, 0x7400_0000), (16, E)],
            24,
            vec![
                ptr(0, &[VmemLoad]),
                ptr(8, &[VmemLoad]),
                ptr(16, &[AtomicRelease]),
            ],
        );
        let mut p = author("gfx1201", &[n0, toucher]);
        p.declare_event_cells(ResourceId(3), 4).unwrap();
        let rec = BindingRecord {
            base: 0,
            template: 1,
            fields: vec![
                BindingField {
                    offset: 0,
                    resource: ResourceId(1),
                    interior: 0,
                },
                BindingField {
                    offset: 8,
                    resource: ResourceId(2),
                    interior: 0,
                },
                BindingField {
                    offset: 16,
                    resource: ResourceId(3),
                    interior: 0,
                },
            ],
            scalars: Vec::new(),
        };
        p.bind_records(ResourceId(0), vec![rec], &SealedTableRegistry::new())
            .unwrap();

        // The arena must be the declared kind and some argument must be bound to it.
        let mut q = p.clone();
        assert!(q
            .admit_lab_pending(0, Some(ResourceId(1)))
            .unwrap_err()
            .contains("not declared BindingRecords"));
        let mut q = author(
            "gfx1201",
            &[unknown(&[(0, A)], 8, vec![ptr(0, &[SmemLoad])])],
        );
        q.kinds.insert(ResourceId(0), ResourceKind::TaskArena);
        assert!(q.admit_lab_pending(0, Some(ResourceId(0))).is_err());
        // A node with no argument bound to the arena.
        let stray = unknown(&[(0, 0x7500_0000)], 8, vec![ptr(0, &[VmemLoad])]);
        let mut r = author("gfx1201", &[stray]);
        r.resources.push(crate::Binding {
            base: A,
            bytes: 0x10000,
        });
        r.kinds
            .insert(ResourceId(1), ResourceKind::BindingRecords(Vec::new()));
        assert!(r
            .admit_lab_pending(0, Some(ResourceId(1)))
            .unwrap_err()
            .contains("no argument of the node is bound"));

        p.admit_lab_pending(0, Some(ResourceId(0))).unwrap();
        let n = &p.nodes[0];
        let on = |r: u32| {
            n.effects
                .iter()
                .find(|e| e.resource == ResourceId(r))
                .unwrap()
        };
        // The arena: its A1 class plus a read, never a write.
        assert!(on(0).has(SmemLoad) && on(0).has(VmemLoad) && !on(0).writes());
        // Plain field resources: load + store through the arena argument.
        for r in [1, 2] {
            assert!(on(r).has(VmemLoad) && on(r).has(VmemStore), "{r}");
            assert_eq!(on(r).arg_offset, 0);
        }
        // The event-cell field: event classes only.
        assert!(on(3).has(LoadAcquire) && on(3).has(AtomicRelease) && !on(3).has(VmemStore));
        assert!(on(3).classes.iter().all(|c| c.is_event_class()));
    }

    /// n0 releases on E and writes D, n1 acquires E and reads D; both A1 Unknown.
    fn lab_group_program() -> KernelProgram {
        let mut p = author(
            "gfx1201",
            &[
                unknown(
                    &[(0, E), (8, D)],
                    16,
                    vec![ptr(0, &[AtomicRelease]), ptr(8, &[VmemStore])],
                ),
                unknown(
                    &[(0, E), (8, D)],
                    16,
                    vec![ptr(0, &[LoadAcquire]), ptr(8, &[VmemLoad])],
                ),
            ],
        );
        p.declare_event_cells(ResourceId(0), 4).unwrap();
        p.admit_lab_pending(0, None).unwrap();
        p.admit_lab_pending(1, None).unwrap();
        p
    }

    #[cfg(feature = "lab")]
    fn edge() -> OrderedEdge {
        OrderedEdge {
            producer: 0,
            consumer: 1,
            resource: ResourceId(1),
            event: ResourceId(0),
        }
    }

    #[test]
    fn a_lab_pending_ordering_admits_the_data_edge_and_the_admission_is_lab_pending() {
        let mut p = lab_group_program();
        // The certificate's ordering is not available: the lab ordering is.
        p.add_coresident_group(0..2, GroupOrdering::lab_pending(vec![edge()]))
            .unwrap();
        let mut g = plan_of(&p);
        assert!(
            g.groups[0].refusals.is_empty(),
            "{:?}",
            g.groups[0].refusals
        );
        let data: Vec<_> = g.groups[0]
            .admitted
            .iter()
            .filter(|(e, _)| e.resource == 1)
            .collect();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0].1, WHY_ORDERED_EDGE);
        g.admit_residency(&p, 0, 10, &[4, 4]).unwrap();
        g.group_admission().unwrap();
        assert!(p.groups[0].ordering.is_pending());
        match g.admission() {
            Admission::LabPending { reasons } => {
                // Both members and the ordering are pending.
                assert_eq!(reasons.len(), 3, "{reasons:?}");
                assert!(reasons
                    .iter()
                    .any(|r| r.contains("ordering edges are not from a task-graph certificate")));
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(g.groups[0].admission(), g.admission());
        assert_eq!(p.admission(), g.admission());
        // A lab-pending ordering's edges are validated like a certificate's.
        let mut bad = lab_group_program();
        let mut e = edge();
        e.producer = 1;
        e.consumer = 0;
        let err = bad
            .add_coresident_group(0..2, GroupOrdering::lab_pending(vec![e]))
            .unwrap_err();
        assert!(err.contains("missing release"), "{err}");
    }

    #[test]
    fn a_lab_pending_node_is_a_node_the_plan_segments_as_pm4() {
        let p = lab_group_program();
        let g = plan_of(&p);
        assert!(g.is_single_segment(2), "segments: {:?}", g.segments);
        // Whereas a HipDirect node splits the plan.
        let mut q = lab_group_program();
        q.nodes[1].lowering = Lowering::HipDirect { reason: "x".into() };
        assert!(!plan_of(&q).is_single_segment(2));
    }

    #[test]
    fn a_lab_pending_node_cannot_join_a_conditional_body() {
        let mut p = lab_group_program();
        let err = p
            .add_conditional(crate::kernel::Conditional {
                flag: ResourceId(0),
                body: 0..1,
                lowering: crate::kernel::CondLowering::CpSkip,
            })
            .unwrap_err();
        assert!(
            err.contains("not PM4") || err.contains("COND_EXEC"),
            "{err}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coresident::{GroupOrdering, OrderedEdge};
    use crate::kernel::tests::{launch, ptr};
    use crate::kernel::{author, MemClass::*};
    use crate::plan::{plan, CacheTable, RowPolicy};
    use crate::ResourceId;

    const E: u64 = 0x7100_0000;
    const D: u64 = 0x7200_0000;

    #[test]
    fn allow_list_matches_whole_fragments_only() {
        for ok in [
            "A1 Unknown (a, b)",
            "descriptor_arena records not bound",
            "indexed_bind witness not consumed",
            "sealed_table witness not consumed",
            "indexed_bind and sealed_table witnesses not consumed",
        ] {
            assert!(is_pending_certificate_reason(ok), "{ok}");
        }
        for bad in [
            "A1 Unknown (a, b) trailing",
            "receipt_missing: x",
            "G8 differential fails",
            "pointer +0 = 0x1 was not probed",
            "descriptor_arena records not bound!",
        ] {
            assert!(!is_pending_certificate_reason(bad), "{bad}");
        }
    }

    #[test]
    fn a_group_of_certified_nodes_and_ordering_is_certified() {
        let mut p = author(
            "gfx1201",
            &[
                launch(
                    &[(0, E), (8, D)],
                    16,
                    vec![ptr(0, &[AtomicRelease]), ptr(8, &[VmemStore])],
                ),
                launch(
                    &[(0, E), (8, D)],
                    16,
                    vec![ptr(0, &[LoadAcquire]), ptr(8, &[VmemLoad])],
                ),
            ],
        );
        p.declare_event_cells(ResourceId(0), 4).unwrap();
        p.add_coresident_group(
            0..2,
            GroupOrdering::from_certificate(vec![OrderedEdge {
                producer: 0,
                consumer: 1,
                resource: ResourceId(1),
                event: ResourceId(0),
            }]),
        )
        .unwrap();
        assert_eq!(p.admission(), Admission::Certified);
        let g = plan(
            &p,
            &CacheTable::for_arch("gfx1201").unwrap(),
            RowPolicy::TierD,
        );
        assert_eq!(g.admission(), Admission::Certified);
        assert_eq!(g.groups[0].admission(), Admission::Certified);
        assert!(!p.groups[0].ordering.is_pending());
    }

    #[test]
    fn receipt_refuses_a_lab_pending_admission_without_the_eager_witness() {
        let lab = Admission::LabPending {
            reasons: vec!["node 0 (k): A1 Unknown (x)".into()],
        };
        let err = lab.receipt(None).unwrap_err();
        assert!(err.contains("eager byte-identity witness"), "{err}");

        let w = DifferentialWitness::compare(b"fixture", b"out", b"out").unwrap();
        let r = lab.receipt(Some(&w)).unwrap();
        assert_eq!(r.admission, LAB_PENDING);
        assert_eq!(r.certified, Some(false));
        assert_eq!(r.pending.as_deref().map(<[_]>::len), Some(1));
        let json = serde_json::to_string(&r).unwrap();
        assert!(
            json.contains(LAB_PENDING) && json.contains("\"certified\":false"),
            "{json}"
        );
        assert!(json.contains(w.eager_sha256()) && json.contains(w.fixture_sha256()));

        let proven = Admission::Certified.receipt(None).unwrap();
        assert_eq!(
            serde_json::to_string(&proven).unwrap(),
            r#"{"admission":"proven"}"#
        );
    }

    #[test]
    fn the_differential_witness_needs_byte_identical_outputs() {
        assert!(DifferentialWitness::compare(b"f", b"abc", b"abd")
            .unwrap_err()
            .contains("byte 2"));
        assert!(DifferentialWitness::compare(b"f", b"abc", b"ab")
            .unwrap_err()
            .contains("common prefix"));
        assert!(DifferentialWitness::compare(b"f", b"", b"").is_err());
        let w = DifferentialWitness::compare(b"f", b"abc", b"abc").unwrap();
        assert_eq!(w.eager_sha256(), w.railgun_sha256());
        assert_ne!(w.fixture_sha256(), w.eager_sha256());
        assert_eq!(w.eager_sha256().len(), 64);
    }

    #[test]
    fn span_lists_wrap_any_admission_and_never_render_without_evidence() {
        assert_eq!(Admission::Certified.with_spans(0), Admission::Certified);
        let once = Admission::Certified.with_spans(1);
        assert_eq!(once.spans(), 1);
        assert_eq!(once.clone().with_spans(2).spans(), 3, "lists add up");
        assert!(once.is_certified());
        let lab = Admission::LabPending {
            reasons: vec!["r".into()],
        }
        .with_spans(1);
        assert!(!lab.is_certified());
        let evidence = CompletenessEvidence {
            producer: "p".into(),
            oracle: "o".into(),
            p0_vs_eager: "e".into(),
        };
        let witness = DifferentialWitness::compare(b"f", b"a", b"a").unwrap();
        for admission in [&once, &lab] {
            assert!(admission.receipt(Some(&witness)).is_err());
            assert!(admission
                .receipt_with_completeness(Some(&witness), None)
                .is_err());
        }
        let r = once
            .receipt_with_completeness(None, Some(&evidence))
            .unwrap();
        assert_eq!(
            serde_json::to_string(&r).unwrap(),
            r#"{"admission":"proven","completeness":"functional (EXPAND_LIVE), not certified","completeness_evidence":{"producer":"p","oracle":"o","p0_vs_eager":"e"}}"#
        );
        let r = lab
            .receipt_with_completeness(Some(&witness), Some(&evidence))
            .unwrap();
        assert_eq!(r.admission, LAB_PENDING);
        assert_eq!(r.completeness, Some(COMPLETENESS_FUNCTIONAL));
    }
}
