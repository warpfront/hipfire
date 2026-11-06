// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Resource kinds of a kernel program beyond a plain allocation (design
//! §8.1): event cells, task arenas and binding-record arenas
//! ([`crate::ResourceKind`]), and the certified effect form of a kernel that
//! reaches its operands through an arena of pointer records.
//!
//! A persistent kernel does not receive its operands as typed kernel
//! arguments: it loads pointers out of per-task records (the packed
//! body-kernarg block), so A1 can attribute those accesses to no argument.
//! The receipt says instead (`descriptor_arenas`, [`crate::kernel::ArenaFact`]):
//! "every address this kernel dereferences through arena argument `a` is a
//! pointer loaded from offset `o` of a record of template `t`, and it is
//! accessed with these classes". The program supplies the records
//! ([`KernelProgram::bind_records`]): each pointer field names a resource and
//! an interior offset, railgun writes it at prepare from the program's own
//! bindings ([`KernelProgram::binding_record_image`]) and it never changes
//! during replay. Binding therefore turns the arena's opaque indirection into
//! ordinary [`RegionEffect`]s on exactly the resources the fields name, and
//! only then is the node PM4-admissible.

use std::collections::{BTreeMap, BTreeSet};

use crate::kernel::{Arg, KernelProgram, Lowering, MemClass, RegionEffect, ARENA_UNBOUND};
use crate::{Binding, BindingField, BindingRecord, ResourceId, ResourceKind, EVENT_CELL_BYTES};

fn event_class_violation(class: MemClass) -> &'static str {
    match class {
        MemClass::VmemStore => "plain store into event cells",
        MemClass::Atomic => "non-commutative atomic on event cells",
        _ => "non-event access class on event cells",
    }
}

impl KernelProgram {
    fn resource_bytes(&self, resource: ResourceId) -> Result<u64, String> {
        self.resources
            .get(resource.0 as usize)
            .map(|b| b.bytes)
            .ok_or_else(|| format!("unknown resource {}", resource.0))
    }

    fn require_undeclared(&self, resource: ResourceId, wanted: &str) -> Result<(), String> {
        match self.kinds.get(&resource) {
            None => Ok(()),
            Some(kind) => Err(format!(
                "resource {} is already declared ({kind:?}); cannot declare it {wanted}",
                resource.0
            )),
        }
    }

    /// Declare `resource` as `cells` event cells ([`EVENT_CELL_BYTES`] each,
    /// design §8.1): device-written only by atomics, observed by acquire
    /// loads. Refused unless the resource exists, holds at least
    /// `cells * 16` bytes and is not otherwise declared, and every node's
    /// effect on it uses only commutative atomics, release atomics, acquire
    /// loads and plain loads: a plain store or a non-commutative atomic is a
    /// "plain store into event cells". On `Err` the program is unchanged.
    pub fn declare_event_cells(&mut self, resource: ResourceId, cells: u32) -> Result<(), String> {
        let bytes = self.resource_bytes(resource)?;
        if cells == 0 {
            return Err(format!("event cells {}: zero cells", resource.0));
        }
        let need = u64::from(cells) * EVENT_CELL_BYTES;
        if bytes < need {
            return Err(format!(
                "event cells {}: {cells} cells need {need} bytes, the resource holds {bytes}",
                resource.0
            ));
        }
        self.require_undeclared(resource, "event cells")?;
        self.check_event_accesses(resource)?;
        self.kinds
            .insert(resource, ResourceKind::EventCells { cells });
        Ok(())
    }

    /// Every node effect on `resource` is an event-cell access.
    pub(crate) fn check_event_accesses(&self, resource: ResourceId) -> Result<(), String> {
        for (i, node) in self.nodes.iter().enumerate() {
            for e in node.effects.iter().filter(|e| e.resource == resource) {
                if let Some(class) = e.classes.iter().find(|c| !c.is_event_class()) {
                    return Err(format!(
                        "node {i} ({}) accesses event cells (resource {}) with {class:?}: {}",
                        node.symbol,
                        resource.0,
                        event_class_violation(*class)
                    ));
                }
            }
        }
        Ok(())
    }

    /// Declare `resource` as a task arena (design §8.1 `TaskDesc` records):
    /// device-rewritten, so nothing about its content is assumed. No node can
    /// turn a loaded value into an address base — A1 attributes accesses only
    /// to explicit pointer arguments and bound descriptor arenas — so the
    /// declaration needs no effect check; it records the resource's role and
    /// refuses a resource that is not there or is already declared.
    pub fn declare_task_arena(&mut self, resource: ResourceId) -> Result<(), String> {
        self.resource_bytes(resource)?;
        self.require_undeclared(resource, "a task arena")?;
        self.kinds.insert(resource, ResourceKind::TaskArena);
        Ok(())
    }

    /// Bind the pointer records of descriptor arena `arena` (§8.1) and
    /// discharge the `descriptor_arena` witness of every node whose receipt
    /// facts name `arena`.
    ///
    /// Refused (the program is unchanged) unless:
    /// - `arena` exists, is not otherwise declared, and no node effect writes
    ///   it (records are immutable during replay);
    /// - records are non-overlapping and inside the arena; every field is
    ///   8-byte aligned, ends inside the arena and does not overlap another
    ///   field of its record ("out-of-bounds descriptor" otherwise);
    /// - every field names an existing resource other than the arena, which is
    ///   not itself a binding-record arena, with `interior` below its size;
    /// - for every node carrying arena facts on `arena`, **every** pointer
    ///   offset those facts say template `t` loads is a declared field of
    ///   **every** record of template `t` ("undeclared arena entry"), and no
    ///   field of a record that a node's facts load lands in an event cell
    ///   with a non-event class;
    /// - every record's template is known to at least one node's facts when any
    ///   node carries facts on `arena` (an arena shared by several kernels
    ///   holds each kernel's own templates, so a template is only required to
    ///   be known by some reader).
    ///
    /// On success each such node gains one [`RegionEffect`] per resource its
    /// loaded fields name (`arg_offset` = the arena argument, classes = the
    /// union over the loaded fields), keeps its arena read effect, and, if the
    /// unbound arena was the only reason it was not PM4, is lowered to PM4
    /// (to `Graph` if the new effects carry a non-commutative atomic).
    pub fn bind_records(
        &mut self,
        arena: ResourceId,
        records: Vec<BindingRecord>,
    ) -> Result<(), String> {
        let arena_bytes = self.resource_bytes(arena)?;
        self.require_undeclared(arena, "binding records")?;
        for (i, node) in self.nodes.iter().enumerate() {
            if node
                .effects
                .iter()
                .any(|e| e.resource == arena && e.writes())
            {
                return Err(format!(
                    "node {i} ({}) writes binding-record arena {}: records are immutable during replay",
                    node.symbol, arena.0
                ));
            }
        }

        let mut extents: Vec<(u64, u64)> = Vec::with_capacity(records.len());
        for rec in &records {
            if rec.base > arena_bytes {
                return Err(format!(
                    "out-of-bounds descriptor: record @{} starts past the {arena_bytes}-byte arena {}",
                    rec.base, arena.0
                ));
            }
            let mut end = rec.base;
            let mut offsets = BTreeSet::new();
            for f in &rec.fields {
                if f.offset % 8 != 0 {
                    return Err(format!(
                        "record @{} field +{}: offset is not 8-byte aligned",
                        rec.base, f.offset
                    ));
                }
                if !offsets.insert(f.offset) {
                    return Err(format!(
                        "record @{} field +{} is declared twice",
                        rec.base, f.offset
                    ));
                }
                let field_end = rec
                    .base
                    .checked_add(u64::from(f.offset))
                    .and_then(|at| at.checked_add(8))
                    .filter(|&e| e <= arena_bytes)
                    .ok_or_else(|| {
                        format!(
                            "out-of-bounds descriptor: record @{} field +{} ends past the {arena_bytes}-byte arena {}",
                            rec.base, f.offset, arena.0
                        )
                    })?;
                end = end.max(field_end);
                self.check_field(arena, rec, f)?;
            }
            extents.push((rec.base, end));
        }
        extents.sort_unstable();
        if let Some(w) = extents.windows(2).find(|w| w[0].1 > w[1].0) {
            return Err(format!(
                "binding records @{}..{} and @{} overlap",
                w[0].0, w[0].1, w[1].0
            ));
        }

        // Effects, per node and arena fact, computed before anything changes.
        let mut known_templates: BTreeSet<u32> = BTreeSet::new();
        let mut readers = false;
        let mut planned: Vec<(usize, usize, Vec<RegionEffect>)> = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            let Some(arena_node) = &node.arena else {
                continue;
            };
            for (k, fact) in arena_node.arenas.iter().enumerate() {
                let is_this_arena = node.args.iter().any(|(offset, arg)| {
                    *offset == fact.arena_arg_offset
                        && matches!(arg, Arg::Resource { resource, interior: 0 } if *resource == arena)
                });
                if !is_this_arena {
                    continue;
                }
                readers = true;
                let mut per_resource: BTreeMap<ResourceId, BTreeSet<MemClass>> = BTreeMap::new();
                for t in &fact.templates {
                    known_templates.insert(t.template);
                    for rec in records.iter().filter(|r| r.template == t.template) {
                        for load in &t.pointer_loads {
                            let Some(field) =
                                rec.fields.iter().find(|f| f.offset == load.record_offset)
                            else {
                                return Err(format!(
                                    "undeclared arena entry: node {i} ({}) loads a pointer at +{} of template {}, which is not a declared field of the record @{}",
                                    node.symbol, load.record_offset, t.template, rec.base
                                ));
                            };
                            if !load.classes.is_empty() {
                                per_resource
                                    .entry(field.resource)
                                    .or_default()
                                    .extend(load.classes.iter().copied());
                            }
                        }
                    }
                }
                let mut effects = Vec::with_capacity(per_resource.len());
                for (resource, classes) in per_resource {
                    if matches!(
                        self.kinds.get(&resource),
                        Some(ResourceKind::EventCells { .. })
                    ) {
                        if let Some(class) = classes.iter().find(|c| !c.is_event_class()) {
                            return Err(format!(
                                "node {i} ({}) accesses event cells (resource {}) through arena records with {class:?}: {}",
                                node.symbol,
                                resource.0,
                                event_class_violation(*class)
                            ));
                        }
                    }
                    effects.push(RegionEffect {
                        arg_offset: fact.arena_arg_offset,
                        resource,
                        classes: classes.into_iter().collect(),
                    });
                }
                planned.push((i, k, effects));
            }
        }
        if readers {
            if let Some(rec) = records
                .iter()
                .find(|r| !known_templates.contains(&r.template))
            {
                return Err(format!(
                    "record @{} has template {}, which no node's descriptor_arena facts know",
                    rec.base, rec.template
                ));
            }
        }

        for (i, k, effects) in planned {
            let node = &mut self.nodes[i];
            node.effects.extend(effects);
            let Some(arena_node) = node.arena.as_mut() else {
                continue;
            };
            arena_node.bound[k] = true;
            if !arena_node.all_bound() {
                continue;
            }
            let Lowering::Conditional { reason } = &node.lowering else {
                continue;
            };
            if !reason.split("; ").any(|p| p == ARENA_UNBOUND) {
                continue;
            }
            if node.indexed.is_none() && node.table.is_none() {
                node.lowering = if node.effects.iter().any(|e| e.has(MemClass::Atomic)) {
                    Lowering::Graph {
                        reason: "atomic without a commutativity certificate".to_owned(),
                    }
                } else {
                    Lowering::Pm4
                };
            } else {
                let rest: Vec<&str> = reason.split("; ").filter(|p| *p != ARENA_UNBOUND).collect();
                node.lowering = Lowering::Conditional {
                    reason: rest.join("; "),
                };
            }
        }
        self.kinds
            .insert(arena, ResourceKind::BindingRecords(records));
        Ok(())
    }

    /// One field's resource checks.
    fn check_field(
        &self,
        arena: ResourceId,
        rec: &BindingRecord,
        f: &BindingField,
    ) -> Result<(), String> {
        let bytes = self.resources.get(f.resource.0 as usize).map(|b| b.bytes);
        let Some(bytes) = bytes else {
            return Err(format!(
                "record @{} field +{}: unknown resource {}",
                rec.base, f.offset, f.resource.0
            ));
        };
        if f.resource == arena {
            return Err(format!(
                "record @{} field +{}: points into its own arena {}",
                rec.base, f.offset, arena.0
            ));
        }
        if matches!(
            self.kinds.get(&f.resource),
            Some(ResourceKind::BindingRecords(_))
        ) {
            return Err(format!(
                "record @{} field +{}: resource {} is itself binding records",
                rec.base, f.offset, f.resource.0
            ));
        }
        if f.interior >= bytes {
            return Err(format!(
                "out-of-bounds descriptor: record @{} field +{} interior {} is past resource {} ({bytes} bytes)",
                rec.base, f.offset, f.interior, f.resource.0
            ));
        }
        Ok(())
    }

    /// The pointer image railgun writes into binding-record arena `arena`:
    /// `(arena byte offset, pointer value)` per field, sorted by offset,
    /// where the value is `base(field.resource) + interior` under
    /// `resources` (the authored [`KernelProgram::resources`], or a window's
    /// live bindings). Checked arithmetic; a field whose interior or whose
    /// slot falls outside the given bindings is an "out-of-bounds descriptor".
    /// The writer of the arena uses exactly this image.
    pub fn binding_record_image(
        &self,
        arena: ResourceId,
        resources: &[Binding],
    ) -> Result<Vec<(u64, u64)>, String> {
        let Some(ResourceKind::BindingRecords(records)) = self.kinds.get(&arena) else {
            return Err(format!(
                "resource {} is not a bound binding-record arena",
                arena.0
            ));
        };
        let arena_bytes = resources
            .get(arena.0 as usize)
            .ok_or_else(|| format!("no binding for arena {}", arena.0))?
            .bytes;
        let mut image = Vec::new();
        for rec in records {
            for f in &rec.fields {
                let at = rec
                    .base
                    .checked_add(u64::from(f.offset))
                    .filter(|at| at.checked_add(8).is_some_and(|e| e <= arena_bytes))
                    .ok_or_else(|| {
                        format!(
                            "out-of-bounds descriptor: record @{} field +{} past the {arena_bytes}-byte arena",
                            rec.base, f.offset
                        )
                    })?;
                let binding = resources
                    .get(f.resource.0 as usize)
                    .ok_or_else(|| format!("no binding for resource {}", f.resource.0))?;
                if f.interior >= binding.bytes {
                    return Err(format!(
                        "out-of-bounds descriptor: record @{} field +{} interior {} is past resource {} ({} bytes)",
                        rec.base, f.offset, f.interior, f.resource.0, binding.bytes
                    ));
                }
                let value = binding
                    .base
                    .checked_add(f.interior)
                    .ok_or_else(|| "pointer overflows".to_owned())?;
                image.push((at, value));
            }
        }
        image.sort_unstable();
        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::tests::{facts, launch, ptr};
    use crate::kernel::{
        author, ArenaFact, ArenaPointerLoad, ArenaTemplate, ConditionalFacts, FactStatus,
        MemClass::*, NodeFacts,
    };
    use std::sync::Arc;

    const A: u64 = 0x7000_0000;
    const B: u64 = 0x7100_0000;
    const C: u64 = 0x7200_0000;
    /// Every launch helper allocation is this big.
    const SIZE: u64 = 0x10000;

    /// Facts of a kernel whose explicit pointer args are `args` and that
    /// carries one descriptor arena at `arena_arg` with `templates`.
    fn arena_facts(
        args: Vec<crate::kernel::ArgFact>,
        segment: u32,
        arena_arg: u32,
        templates: Vec<ArenaTemplate>,
    ) -> Arc<NodeFacts> {
        let mut f = (*facts(args, segment)).clone();
        f.facts.status = FactStatus::Conditional(ConditionalFacts {
            guards: Vec::new(),
            indexed: Vec::new(),
            tables: Vec::new(),
            arenas: vec![ArenaFact {
                arena_arg_offset: arena_arg,
                templates,
            }],
            witnesses: vec!["descriptor_arena".into()],
            case_digest: "ab".repeat(32),
        });
        Arc::new(f)
    }

    fn template(template: u32, loads: &[(u32, &[crate::kernel::MemClass])]) -> ArenaTemplate {
        ArenaTemplate {
            template,
            pointer_loads: loads
                .iter()
                .map(|&(record_offset, classes)| ArenaPointerLoad {
                    record_offset,
                    classes: classes.to_vec(),
                })
                .collect(),
        }
    }

    /// Resources: 0 = arena (read through arg 0), 1 = X, 2 = Y. One node
    /// whose arena facts load template 7's pointers at +0 (reads) and +8
    /// (stores).
    fn arena_program() -> KernelProgram {
        let args = vec![ptr(0, &[SmemLoad])];
        let mut l = launch(&[(0, A)], 8, args.clone());
        l.facts = Ok(arena_facts(
            args,
            8,
            0,
            vec![template(7, &[(0, &[VmemLoad]), (8, &[VmemStore])])],
        ));
        author("gfx1201", &[l])
    }

    /// `arena_program` plus X and Y as resources 1 and 2 (touched by a second
    /// plain node so the program knows them).
    fn arena_program_with_targets() -> KernelProgram {
        let args = vec![ptr(0, &[SmemLoad])];
        let mut arena = launch(&[(0, A)], 8, args.clone());
        arena.facts = Ok(arena_facts(
            args,
            8,
            0,
            vec![template(7, &[(0, &[VmemLoad]), (8, &[VmemStore])])],
        ));
        let toucher = launch(
            &[(0, B), (8, C)],
            16,
            vec![ptr(0, &[VmemLoad]), ptr(8, &[VmemLoad])],
        );
        author("gfx1201", &[arena, toucher])
    }

    fn rec(base: u64, template: u32, fields: &[(u32, u32, u64)]) -> BindingRecord {
        BindingRecord {
            base,
            template,
            fields: fields
                .iter()
                .map(|&(offset, resource, interior)| BindingField {
                    offset,
                    resource: ResourceId(resource),
                    interior,
                })
                .collect(),
        }
    }

    #[test]
    fn event_cells_refuse_a_plain_store_and_a_noncommutative_atomic() {
        let ptrs = |class| {
            author(
                "gfx1201",
                &[
                    launch(&[(0, A)], 8, vec![ptr(0, &[AtomicRelease])]),
                    launch(&[(0, A)], 8, vec![ptr(0, &[class])]),
                ],
            )
        };
        let mut p = ptrs(VmemStore);
        let before = p.clone();
        let err = p.declare_event_cells(ResourceId(0), 4).unwrap_err();
        assert!(err.contains("plain store into event cells"), "{err}");
        assert_eq!(p, before, "an Err leaves the program unchanged");

        let mut p = ptrs(Atomic);
        let err = p.declare_event_cells(ResourceId(0), 4).unwrap_err();
        assert!(err.contains("non-commutative atomic"), "{err}");

        let mut p = ptrs(LoadAcquire);
        p.declare_event_cells(ResourceId(0), 4).unwrap();
        assert_eq!(
            p.kinds[&ResourceId(0)],
            ResourceKind::EventCells { cells: 4 }
        );
        assert!(p.declare_event_cells(ResourceId(0), 4).is_err(), "twice");
        assert!(p.declare_task_arena(ResourceId(0)).is_err(), "redeclared");
    }

    #[test]
    fn event_cells_need_room_and_an_existing_resource() {
        let mut p = author(
            "gfx1201",
            &[launch(&[(0, A)], 8, vec![ptr(0, &[AtomicRelease])])],
        );
        let cells = (SIZE / EVENT_CELL_BYTES) as u32;
        assert!(p.declare_event_cells(ResourceId(0), cells + 1).is_err());
        assert!(p.declare_event_cells(ResourceId(0), 0).is_err());
        assert!(p.declare_event_cells(ResourceId(9), 1).is_err());
        p.declare_event_cells(ResourceId(0), cells).unwrap();
        assert!(p.declare_task_arena(ResourceId(9)).is_err());
    }

    #[test]
    fn task_arena_is_recorded() {
        let mut p = author(
            "gfx1201",
            &[launch(&[(0, A)], 8, vec![ptr(0, &[VmemStore])])],
        );
        p.declare_task_arena(ResourceId(0)).unwrap();
        assert_eq!(p.kinds[&ResourceId(0)], ResourceKind::TaskArena);
    }

    #[test]
    fn an_unbound_arena_node_is_conditional_and_arena_arg_must_be_a_base() {
        let p = arena_program();
        assert_eq!(
            p.nodes[0].lowering,
            Lowering::Conditional {
                reason: ARENA_UNBOUND.into()
            }
        );
        assert!(p.nodes[0].arena.is_some());
        // The only effect is the arena read; the fields' targets are unknown.
        assert_eq!(p.nodes[0].effects.len(), 1);
        // No window witness stands in for `bind_records`.
        let assumed = p.assume_witnesses(&["indexed_bind", "sealed_table", "descriptor_arena"]);
        assert!(matches!(
            assumed.nodes[0].lowering,
            Lowering::Conditional { .. }
        ));

        // An interior arena pointer is refused (the arena must be a base).
        let args = vec![ptr(0, &[SmemLoad])];
        let mut l = launch(&[(0, A + 0x40)], 8, args.clone());
        l.facts = Ok(arena_facts(
            args,
            8,
            0,
            vec![template(7, &[(0, &[VmemLoad])])],
        ));
        let p = author("gfx1201", &[l]);
        assert!(
            matches!(&p.nodes[0].lowering, Lowering::HipDirect { reason } if reason.contains("interior 0")),
            "{:?}",
            p.nodes[0].lowering
        );
    }

    #[test]
    fn bound_arena_node_is_pm4_with_effects_on_exactly_the_field_resources() {
        let mut p = arena_program_with_targets();
        // resources: 0 = A (arena), 1 = B (X), 2 = C (Y)
        p.bind_records(
            ResourceId(0),
            vec![
                rec(0, 7, &[(0, 1, 0x100), (8, 2, 0x20)]),
                rec(64, 7, &[(0, 2, 0x8), (8, 1, 0x10)]),
            ],
        )
        .unwrap();
        let n = &p.nodes[0];
        assert_eq!(n.lowering, Lowering::Pm4);
        assert!(n.arena.as_ref().unwrap().all_bound());
        // The arena read effect stays; one merged effect per field resource.
        let mut by_resource: Vec<(u32, Vec<crate::kernel::MemClass>)> = n
            .effects
            .iter()
            .map(|e| (e.resource.0, e.classes.clone()))
            .collect();
        by_resource.sort();
        assert_eq!(
            by_resource,
            vec![
                (0, vec![SmemLoad]),
                (1, vec![VmemLoad, VmemStore]),
                (2, vec![VmemLoad, VmemStore]),
            ]
        );
        assert!(n.effects[1..].iter().all(|e| e.arg_offset == 0));
        assert!(matches!(
            p.kinds[&ResourceId(0)],
            ResourceKind::BindingRecords(_)
        ));
        // Bound twice: refused.
        assert!(p.bind_records(ResourceId(0), Vec::new()).is_err());
    }

    #[test]
    fn undeclared_arena_entry_is_refused() {
        let mut p = arena_program_with_targets();
        let before = p.clone();
        // Template 7's facts load +0 and +8; this record declares only +0.
        let err = p
            .bind_records(ResourceId(0), vec![rec(0, 7, &[(0, 1, 0)])])
            .unwrap_err();
        assert!(err.contains("undeclared arena entry"), "{err}");
        assert_eq!(p, before);

        // A record of a template no node's facts know.
        let err = p
            .bind_records(ResourceId(0), vec![rec(0, 9, &[(0, 1, 0), (8, 2, 0)])])
            .unwrap_err();
        assert!(err.contains("template 9"), "{err}");
        assert_eq!(p, before);

        // An extra declared field that no facts load is fine.
        let mut ok = p.clone();
        ok.bind_records(
            ResourceId(0),
            vec![rec(0, 7, &[(0, 1, 0), (8, 2, 0), (16, 1, 0)])],
        )
        .unwrap();
        assert_eq!(ok.nodes[0].lowering, Lowering::Pm4);
    }

    #[test]
    fn out_of_bounds_descriptors_are_refused() {
        let mut p = arena_program_with_targets();
        let before = p.clone();
        // A field past the arena end (arena = SIZE bytes).
        let err = p
            .bind_records(
                ResourceId(0),
                vec![rec(SIZE - 8, 7, &[(0, 1, 0), (8, 2, 0)])],
            )
            .unwrap_err();
        assert!(err.contains("out-of-bounds descriptor"), "{err}");
        // An interior past the pointee.
        let err = p
            .bind_records(ResourceId(0), vec![rec(0, 7, &[(0, 1, SIZE), (8, 2, 0)])])
            .unwrap_err();
        assert!(err.contains("out-of-bounds descriptor"), "{err}");
        // Misaligned, duplicate, self-pointing, unknown, overlapping.
        assert!(p
            .bind_records(ResourceId(0), vec![rec(0, 7, &[(4, 1, 0)])])
            .is_err());
        assert!(p
            .bind_records(ResourceId(0), vec![rec(0, 7, &[(0, 1, 0), (0, 2, 0)])])
            .is_err());
        assert!(p
            .bind_records(ResourceId(0), vec![rec(0, 7, &[(0, 0, 0)])])
            .is_err());
        assert!(p
            .bind_records(ResourceId(0), vec![rec(0, 7, &[(0, 9, 0)])])
            .is_err());
        assert!(p
            .bind_records(
                ResourceId(0),
                vec![
                    rec(0, 7, &[(0, 1, 0), (8, 2, 0)]),
                    rec(8, 7, &[(0, 1, 0), (8, 2, 0)])
                ]
            )
            .is_err());
        assert_eq!(p, before);
    }

    #[test]
    fn an_arena_a_node_writes_cannot_be_bound() {
        let args = vec![ptr(0, &[SmemLoad])];
        let mut arena = launch(&[(0, A)], 8, args.clone());
        arena.facts = Ok(arena_facts(
            args,
            8,
            0,
            vec![template(7, &[(0, &[VmemLoad])])],
        ));
        let writer = launch(&[(0, A)], 8, vec![ptr(0, &[VmemStore])]);
        let toucher = launch(&[(0, B)], 8, vec![ptr(0, &[VmemLoad])]);
        let mut p = author("gfx1201", &[arena, writer, toucher]);
        let before = p.clone();
        let err = p
            .bind_records(ResourceId(0), vec![rec(0, 7, &[(0, 1, 0)])])
            .unwrap_err();
        assert!(err.contains("immutable during replay"), "{err}");
        assert_eq!(p, before);
    }

    #[test]
    fn arena_fields_into_event_cells_need_event_classes() {
        let mut p = arena_program_with_targets();
        // Resource 1 as event cells: template 7 stores through field +8 only
        // to resource 2, but loads +0 (VmemLoad, an event class) from 1.
        p.declare_event_cells(ResourceId(1), 4).unwrap();
        p.bind_records(ResourceId(0), vec![rec(0, 7, &[(0, 1, 0), (8, 2, 0)])])
            .unwrap();
        // Pointing the stored field at the event cells is a plain store.
        let mut q = arena_program_with_targets();
        q.declare_event_cells(ResourceId(1), 4).unwrap();
        let err = q
            .bind_records(ResourceId(0), vec![rec(0, 7, &[(0, 2, 0), (8, 1, 0)])])
            .unwrap_err();
        assert!(err.contains("plain store into event cells"), "{err}");
    }

    #[test]
    fn binding_record_image_is_base_plus_interior_from_the_given_bindings() {
        let mut p = arena_program_with_targets();
        p.bind_records(
            ResourceId(0),
            vec![
                rec(64, 7, &[(0, 2, 0x8), (8, 1, 0x10)]),
                rec(0, 7, &[(0, 1, 0x100), (8, 2, 0x20)]),
            ],
        )
        .unwrap();
        let image = p.binding_record_image(ResourceId(0), &p.resources).unwrap();
        // Resource 1 is B's allocation, 2 is C's (launch bases are 4 KiB aligned).
        assert_eq!(
            image,
            vec![(0, B + 0x100), (8, C + 0x20), (64, C + 0x8), (72, B + 0x10),]
        );
        // A window's live bindings re-base every field.
        let live: Vec<Binding> = p
            .resources
            .iter()
            .map(|b| Binding {
                base: b.base + 0x1_0000_0000,
                bytes: b.bytes,
            })
            .collect();
        let moved = p.binding_record_image(ResourceId(0), &live).unwrap();
        assert_eq!(moved[0], (0, B + 0x1_0000_0100));
        // Smaller live bindings refuse an interior outside them.
        let mut small = p.resources.clone();
        small[1].bytes = 0x80;
        let err = p.binding_record_image(ResourceId(0), &small).unwrap_err();
        assert!(err.contains("out-of-bounds descriptor"), "{err}");
        // A resource that is not bound records has no image.
        assert!(p.binding_record_image(ResourceId(1), &p.resources).is_err());
        // Address arithmetic is checked.
        let mut huge = p.resources.clone();
        huge[1].base = u64::MAX - 8;
        assert!(p.binding_record_image(ResourceId(0), &huge).is_err());
    }

    #[test]
    fn a_receipt_record_parses_into_arena_facts_and_fails_closed() {
        use crate::kernel::KernelFacts;
        use crate::sealed_table::tests as st;
        use railgun_corpus::receipt::{
            ArenaPointerRecord, ArenaTemplateRecord, DescriptorArenaRecord,
            WITNESS_DESCRIPTOR_ARENA,
        };
        let mut k = st::record_with(Vec::new());
        k.descriptor_arenas = vec![DescriptorArenaRecord {
            arena_arg_offset: 0,
            templates: vec![ArenaTemplateRecord {
                template: 3,
                pointer_loads: vec![ArenaPointerRecord {
                    record_offset: 8,
                    classes: vec!["load_acquire".into(), "vmem_load".into()],
                }],
            }],
        }];
        k.required_witnesses = k.implied_witnesses();
        assert_eq!(k.required_witnesses, vec![WITNESS_DESCRIPTOR_ARENA]);
        let FactStatus::Conditional(c) = KernelFacts::from_record(&k).unwrap().status else {
            panic!("conditional")
        };
        assert_eq!(c.witnesses, vec![WITNESS_DESCRIPTOR_ARENA]);
        assert_eq!(c.arenas.len(), 1);
        assert_eq!(c.arenas[0].arena_arg_offset, 0);
        assert_eq!(c.arenas[0].templates[0].template, 3);
        assert_eq!(
            c.arenas[0].templates[0].pointer_loads,
            vec![ArenaPointerLoad {
                record_offset: 8,
                classes: vec![VmemLoad, LoadAcquire],
            }]
        );

        // The arena argument must be an 8-byte pointer argument.
        let mut bad = k.clone();
        bad.descriptor_arenas[0].arena_arg_offset = 16;
        assert!(KernelFacts::from_record(&bad).is_err());
        // An unknown memory class.
        let mut bad = k.clone();
        bad.descriptor_arenas[0].templates[0].pointer_loads[0].classes = vec!["bogus".into()];
        assert!(KernelFacts::from_record(&bad).is_err());
        // The required witnesses must be exactly what the case implies.
        let mut bad = k.clone();
        bad.required_witnesses = vec!["sealed_table".into()];
        assert!(KernelFacts::from_record(&bad).is_err());
        // Two arenas on one argument.
        let mut bad = k.clone();
        bad.descriptor_arenas.push(bad.descriptor_arenas[0].clone());
        assert!(KernelFacts::from_record(&bad).is_err());
    }

    #[test]
    fn arena_fields_may_name_allocations_no_kernarg_reaches() {
        let mut p = arena_program();
        assert_eq!(
            p.resources.len(),
            1,
            "only the arena is an explicit argument"
        );
        let x = p
            .resource_for(Binding {
                base: B,
                bytes: SIZE,
            })
            .unwrap();
        let y = p
            .resource_for(Binding {
                base: C,
                bytes: SIZE,
            })
            .unwrap();
        assert_eq!((x, y), (ResourceId(1), ResourceId(2)));
        assert_eq!(
            p.resource_for(Binding {
                base: B,
                bytes: SIZE
            }),
            Ok(x),
            "same allocation, same id"
        );
        let err = p
            .resource_for(Binding {
                base: B + 8,
                bytes: SIZE,
            })
            .unwrap_err();
        assert!(err.contains("overlaps resource 1"), "{err}");
        assert!(p.resource_for(Binding { base: B, bytes: 0 }).is_err());
        p.bind_records(ResourceId(0), vec![rec(0, 7, &[(0, 1, 0), (8, 2, 16)])])
            .unwrap();
        assert!(p.nodes[0].lowering.is_pm4(), "{:?}", p.nodes[0].lowering);
        assert_eq!(
            p.binding_record_image(ResourceId(0), &p.resources).unwrap(),
            vec![(0, B), (8, C + 16)]
        );
    }
}
