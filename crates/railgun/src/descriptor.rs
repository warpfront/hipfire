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

use crate::kernel::{
    ArenaSealedBinding, Arg, KernelProgram, Lowering, MemClass, RegionEffect, SealedTableFact,
    TableBound, ARENA_UNBOUND,
};
use crate::sealed_table::{overlaps, SealedTableError, SealedTableRegistry, MAX_BOUND};
use crate::{
    Binding, BindingField, BindingRecord, BindingRecordImage, ResourceId, ResourceKind,
    EVENT_CELL_BYTES,
};

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
    ///
    /// **Sealed tables (design D.2).** A template whose facts carry sealed
    /// tables (`ArenaTemplate::sealed`: the body loads an expert id from a
    /// device-written id table, guards it below a bound and reads through a
    /// pointer loaded from a loader-written pointer table) additionally
    /// requires, for **every** record of the template, that the table field
    /// names a resource `sealed` holds a seal for (loader-sealed immutable
    /// content whose every entry points into an immutable allocation; see
    /// [`SealedTableRegistry::seal`]), that the bound is a constant or a
    /// scalar **declared in the record** ([`BindingRecord::scalars`], written
    /// by railgun like the pointer fields) at most the table's entry count,
    /// and that no node writes the table or any pointee allocation. The node
    /// then also gains a read effect on the table and on every pointee that
    /// is a program resource, records the bound tables
    /// ([`ArenaNode::sealed`]) and stays `Conditional` until a window's
    /// `sealed_table` witness ([`SealedTableRegistry::bind_sealed_table`])
    /// re-verifies generation, content and the full-program no-writer
    /// closure and holds the registry lease for the submission.
    pub fn bind_records(
        &mut self,
        arena: ResourceId,
        records: Vec<BindingRecord>,
        sealed: &SealedTableRegistry,
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
            let mut scalar_offsets = BTreeSet::new();
            for &(offset, _) in &rec.scalars {
                if offset % 4 != 0 {
                    return Err(format!(
                        "record @{} scalar +{offset}: offset is not 4-byte aligned",
                        rec.base
                    ));
                }
                if !scalar_offsets.insert(offset) {
                    return Err(format!(
                        "record @{} scalar +{offset} is declared twice",
                        rec.base
                    ));
                }
                if rec
                    .fields
                    .iter()
                    .any(|f| offset < f.offset + 8 && f.offset < offset + 4)
                {
                    return Err(format!(
                        "record @{} scalar +{offset} overlaps a pointer field",
                        rec.base
                    ));
                }
                let scalar_end = rec
                    .base
                    .checked_add(u64::from(offset))
                    .and_then(|at| at.checked_add(4))
                    .filter(|&e| e <= arena_bytes)
                    .ok_or_else(|| {
                        format!(
                            "out-of-bounds descriptor: record @{} scalar +{offset} ends past the {arena_bytes}-byte arena {}",
                            rec.base, arena.0
                        )
                    })?;
                end = end.max(scalar_end);
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
        let mut planned: Vec<(usize, usize, Vec<RegionEffect>, Vec<ArenaSealedBinding>)> =
            Vec::new();
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
                let mut bindings: Vec<ArenaSealedBinding> = Vec::new();
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
                        for s in &t.sealed {
                            let Some(d) =
                                self.discharge_sealed(i, k, t.template, rec, s, sealed)?
                            else {
                                continue;
                            };
                            // The table read and the reads of every pointee.
                            per_resource
                                .entry(d.resource)
                                .or_default()
                                .extend(s.entries.iter().map(|e| e.class));
                            for p in &d.pointees {
                                per_resource
                                    .entry(*p)
                                    .or_default()
                                    .extend(s.accesses.iter().map(|a| a.class));
                            }
                            match bindings.iter_mut().find(|b| {
                                b.template == d.template
                                    && b.table_arg_offset == d.table_arg_offset
                                    && b.resource == d.resource
                            }) {
                                Some(b) => b.bound = b.bound.max(d.bound),
                                None => bindings.push(d),
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
                planned.push((i, k, effects, bindings));
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

        // Sealed content is immutable: no effect this bind adds may write a
        // table or pointee, whether it is sealed by this bind or an earlier
        // one (the per-record checks above only saw the effects already on
        // the nodes).
        let mut protected: BTreeMap<ResourceId, &str> = BTreeMap::new();
        for b in self
            .nodes
            .iter()
            .filter_map(|n| n.arena.as_ref())
            .flat_map(|a| a.sealed.iter())
            .chain(planned.iter().flat_map(|(_, _, _, b)| b.iter()))
        {
            protected.insert(b.resource, "table allocation");
            for p in &b.pointees {
                protected.entry(*p).or_insert("pointee allocation");
            }
        }
        for (i, _, effects, _) in &planned {
            if let Some(e) = effects
                .iter()
                .find(|e| e.writes() && protected.contains_key(&e.resource))
            {
                return Err(format!(
                    "node {i} ({}) writes the sealed {} (resource {}) through the arena records",
                    self.nodes[*i].symbol, protected[&e.resource], e.resource.0
                ));
            }
        }
        for (i, k, effects, bindings) in planned {
            let node = &mut self.nodes[i];
            node.effects.extend(effects);
            let Some(arena_node) = node.arena.as_mut() else {
                continue;
            };
            arena_node.bound[k] = true;
            arena_node.sealed.extend(bindings);
            if !arena_node.all_bound() {
                continue;
            }
            let Lowering::Conditional { reason } = &node.lowering else {
                continue;
            };
            if !reason.split("; ").any(|p| p == ARENA_UNBOUND) {
                continue;
            }
            if node.indexed.is_none() && !node.needs_sealed_table_witness() {
                node.lowering = if node.effects.iter().any(|e| e.has(MemClass::Atomic)) {
                    Lowering::Graph {
                        reason: "atomic without a commutativity certificate".to_owned(),
                    }
                } else {
                    Lowering::Pm4
                };
            } else {
                let mut rest: Vec<&str> =
                    reason.split("; ").filter(|p| *p != ARENA_UNBOUND).collect();
                // Sealed tables of the bound templates are discharged by a
                // window's `sealed_table` witness, not by the bind alone.
                if node.needs_sealed_table_witness()
                    && !rest.iter().any(|p| p.contains("sealed_table witness"))
                {
                    rest.push("sealed_table witness not consumed");
                }
                node.lowering = Lowering::Conditional {
                    reason: rest.join("; "),
                };
            }
        }
        self.kinds
            .insert(arena, ResourceKind::BindingRecords(records));
        Ok(())
    }

    /// Discharge sealed table `s` of template `template` for one record `rec`
    /// (design D.2: bounded indices into a sealed immutable source bank).
    /// `Ok(None)` is a record whose bound is zero: the certified guard
    /// `index < 0` is unsatisfiable, so no entry is ever loaded and nothing
    /// is required of the table.
    ///
    /// Otherwise it is refused unless: the table pointer field exists with
    /// interior 0; the bound is a constant or a **declared scalar of this
    /// record** (a byte nobody declared is not a bound railgun knows); the
    /// table field's resource is exactly the allocation `sealed` holds a
    /// seal for (the registry's own identity check, never a copy of it), the
    /// bound is at most the sealed entry count, and a table with null entries
    /// is read only by a null-guarded certificate; and nothing in the program
    /// can write the table or any pointee allocation — a pointee that is a
    /// program resource must equal that allocation exactly, none partially
    /// overlaps one, and no node effect writes any of them.
    fn discharge_sealed(
        &self,
        node: usize,
        arena_fact: usize,
        template: u32,
        rec: &BindingRecord,
        s: &SealedTableFact,
        sealed: &SealedTableRegistry,
    ) -> Result<Option<ArenaSealedBinding>, String> {
        let who = format!(
            "node {node} ({}) sealed table +{} of template {} record @{}",
            self.nodes[node].symbol, s.table_arg_offset, template, rec.base
        );
        let Some(field) = rec.fields.iter().find(|f| f.offset == s.table_arg_offset) else {
            return Err(format!(
                "undeclared arena entry: {who}: the table pointer is not a declared field"
            ));
        };
        if field.interior != 0 {
            return Err(format!(
                "{who}: the table field points {} bytes into resource {}, not at its base",
                field.interior, field.resource.0
            ));
        }
        let bound = match s.bound {
            TableBound::Const(c) => c,
            TableBound::Kernarg(at) => rec
                .scalars
                .iter()
                .find(|(o, _)| *o == at)
                .map(|(_, v)| *v)
                .ok_or_else(|| {
                    format!("undeclared arena entry: {who}: the bound at +{at} is not a declared scalar of the record")
                })?,
        };
        if bound == 0 {
            return Ok(None);
        }
        if bound > MAX_BOUND {
            return Err(SealedTableError::BoundTooLarge {
                node,
                table_arg: s.table_arg_offset,
                bound,
            }
            .to_string());
        }
        let resource = field.resource;
        let binding = self.resources[resource.0 as usize];
        let table = sealed.table_at(binding.base).ok_or_else(|| {
            SealedTableError::NotSealed(format!(
                "no sealed table at {:#x}+{} ({who})",
                binding.base, binding.bytes
            ))
            .to_string()
        })?;
        if table.allocation != binding {
            return Err(SealedTableError::WrongAllocation(format!(
                "{who}: resource {} is {:#x}+{}, the table sealed at that base is {:#x}+{}",
                resource.0,
                binding.base,
                binding.bytes,
                table.allocation.base,
                table.allocation.bytes
            ))
            .to_string());
        }
        if table.has_null_entries && !s.null_guarded {
            return Err(SealedTableError::NullEntryUnguarded {
                node,
                table_arg: s.table_arg_offset,
            }
            .to_string());
        }
        if bound > table.entries {
            return Err(SealedTableError::BoundExceedsEntries {
                node,
                table_arg: s.table_arg_offset,
                bound,
                entries: table.entries,
            }
            .to_string());
        }
        // The no-writer closure over the program's resources and effects.
        let plain = |r: ResourceId, what: &str| match self.kinds.get(&r) {
            None => Ok(()),
            Some(kind) => Err(format!(
                "{who}: the {what} (resource {}) is declared {kind:?}, not an immutable allocation",
                r.0
            )),
        };
        plain(resource, "table")?;
        let mut pointees: Vec<ResourceId> = Vec::new();
        for (ix, b) in self.resources.iter().enumerate() {
            let r = ResourceId(ix as u32);
            if r != resource && overlaps(*b, table.allocation) {
                return Err(SealedTableError::Closure(format!(
                    "{who}: resource {} {:#x}+{} overlaps the sealed table allocation {:#x}+{}",
                    r.0, b.base, b.bytes, table.allocation.base, table.allocation.bytes
                ))
                .to_string());
            }
            for p in &table.pointees {
                if !overlaps(*b, p.allocation) {
                    continue;
                }
                if *b != p.allocation {
                    return Err(SealedTableError::Closure(format!(
                        "{who}: resource {} {:#x}+{} partially overlaps the sealed pointee allocation {:#x}+{}",
                        r.0, b.base, b.bytes, p.allocation.base, p.allocation.bytes
                    ))
                    .to_string());
                }
                plain(r, "pointee")?;
                if !pointees.contains(&r) {
                    pointees.push(r);
                }
            }
        }
        for (j, other) in self.nodes.iter().enumerate() {
            if let Some(e) = other
                .effects
                .iter()
                .find(|e| e.writes() && (e.resource == resource || pointees.contains(&e.resource)))
            {
                return Err(SealedTableError::Closure(format!(
                    "{who}: node {j} ({}) writes the sealed {} (resource {})",
                    other.symbol,
                    if e.resource == resource {
                        "table allocation"
                    } else {
                        "pointee allocation"
                    },
                    e.resource.0
                ))
                .to_string());
            }
        }
        Ok(Some(ArenaSealedBinding {
            arena: arena_fact,
            template,
            table_arg_offset: s.table_arg_offset,
            resource,
            bound,
            null_guarded: s.null_guarded,
            allocation: table.allocation,
            entries: table.entries,
            generation: table.generation,
            content_sha256: table.content_sha256,
            pointees,
            accesses: {
                let mut a: Vec<MemClass> = s.accesses.iter().map(|a| a.class).collect();
                a.sort();
                a.dedup();
                a
            },
        }))
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

    /// What railgun writes into binding-record arena `arena`
    /// ([`BindingRecordImage`]): `(arena byte offset, pointer value)` per
    /// pointer field, where the value is `base(field.resource) + interior`
    /// under `resources` (the authored [`KernelProgram::resources`], or a
    /// window's live bindings), and `(arena byte offset, value)` per declared
    /// `u32` scalar; both sorted by offset. Checked arithmetic; a field or
    /// scalar whose interior or whose slot falls outside the given bindings,
    /// or that overlaps another write, is an "out-of-bounds descriptor". The
    /// writer of the arena uses exactly this image.
    pub fn binding_record_image(
        &self,
        arena: ResourceId,
        resources: &[Binding],
    ) -> Result<BindingRecordImage, String> {
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
        let mut image = BindingRecordImage::default();
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
                image.pointers.push((at, value));
            }
            for &(offset, value) in &rec.scalars {
                let at = rec
                    .base
                    .checked_add(u64::from(offset))
                    .filter(|at| at.checked_add(4).is_some_and(|e| e <= arena_bytes))
                    .ok_or_else(|| {
                        format!(
                            "out-of-bounds descriptor: record @{} scalar +{offset} past the {arena_bytes}-byte arena",
                            rec.base
                        )
                    })?;
                image.scalars.push((at, value));
            }
        }
        image.pointers.sort_unstable();
        image.scalars.sort_unstable();
        let writes = image.writes();
        if let Some(w) = writes
            .windows(2)
            .find(|w| w[0].0 + w[0].1.len() as u64 > w[1].0)
        {
            return Err(format!(
                "out-of-bounds descriptor: the write at +{} overlaps the one at +{}",
                w[0].0, w[1].0
            ));
        }
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
            task_protocol: None,
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
            sealed: Vec::new(),
            wg_footprints: Vec::new(),
            hidden_grid: Vec::new(),
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
            scalars: Vec::new(),
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
            &SealedTableRegistry::new(),
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
        assert!(p
            .bind_records(ResourceId(0), Vec::new(), &SealedTableRegistry::new())
            .is_err());
    }

    #[test]
    fn undeclared_arena_entry_is_refused() {
        let mut p = arena_program_with_targets();
        let before = p.clone();
        // Template 7's facts load +0 and +8; this record declares only +0.
        let err = p
            .bind_records(
                ResourceId(0),
                vec![rec(0, 7, &[(0, 1, 0)])],
                &SealedTableRegistry::new(),
            )
            .unwrap_err();
        assert!(err.contains("undeclared arena entry"), "{err}");
        assert_eq!(p, before);

        // A record of a template no node's facts know.
        let err = p
            .bind_records(
                ResourceId(0),
                vec![rec(0, 9, &[(0, 1, 0), (8, 2, 0)])],
                &SealedTableRegistry::new(),
            )
            .unwrap_err();
        assert!(err.contains("template 9"), "{err}");
        assert_eq!(p, before);

        // An extra declared field that no facts load is fine.
        let mut ok = p.clone();
        ok.bind_records(
            ResourceId(0),
            vec![rec(0, 7, &[(0, 1, 0), (8, 2, 0), (16, 1, 0)])],
            &SealedTableRegistry::new(),
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
                &SealedTableRegistry::new(),
            )
            .unwrap_err();
        assert!(err.contains("out-of-bounds descriptor"), "{err}");
        // An interior past the pointee.
        let err = p
            .bind_records(
                ResourceId(0),
                vec![rec(0, 7, &[(0, 1, SIZE), (8, 2, 0)])],
                &SealedTableRegistry::new(),
            )
            .unwrap_err();
        assert!(err.contains("out-of-bounds descriptor"), "{err}");
        // Misaligned, duplicate, self-pointing, unknown, overlapping.
        let none = SealedTableRegistry::new();
        assert!(p
            .bind_records(ResourceId(0), vec![rec(0, 7, &[(4, 1, 0)])], &none)
            .is_err());
        assert!(p
            .bind_records(
                ResourceId(0),
                vec![rec(0, 7, &[(0, 1, 0), (0, 2, 0)])],
                &none
            )
            .is_err());
        assert!(p
            .bind_records(ResourceId(0), vec![rec(0, 7, &[(0, 0, 0)])], &none)
            .is_err());
        assert!(p
            .bind_records(ResourceId(0), vec![rec(0, 7, &[(0, 9, 0)])], &none)
            .is_err());
        assert!(p
            .bind_records(
                ResourceId(0),
                vec![
                    rec(0, 7, &[(0, 1, 0), (8, 2, 0)]),
                    rec(8, 7, &[(0, 1, 0), (8, 2, 0)])
                ],
                &none
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
            .bind_records(
                ResourceId(0),
                vec![rec(0, 7, &[(0, 1, 0)])],
                &SealedTableRegistry::new(),
            )
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
        p.bind_records(
            ResourceId(0),
            vec![rec(0, 7, &[(0, 1, 0), (8, 2, 0)])],
            &SealedTableRegistry::new(),
        )
        .unwrap();
        // Pointing the stored field at the event cells is a plain store.
        let mut q = arena_program_with_targets();
        q.declare_event_cells(ResourceId(1), 4).unwrap();
        let err = q
            .bind_records(
                ResourceId(0),
                vec![rec(0, 7, &[(0, 2, 0), (8, 1, 0)])],
                &SealedTableRegistry::new(),
            )
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
            &SealedTableRegistry::new(),
        )
        .unwrap();
        let image = p.binding_record_image(ResourceId(0), &p.resources).unwrap();
        // Resource 1 is B's allocation, 2 is C's (launch bases are 4 KiB aligned).
        assert_eq!(
            image.pointers,
            vec![(0, B + 0x100), (8, C + 0x20), (64, C + 0x8), (72, B + 0x10),]
        );
        assert!(image.scalars.is_empty());
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
        assert_eq!(moved.pointers[0], (0, B + 0x1_0000_0100));
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
                sealed_tables: vec![],
                wg_footprints: vec![],
                hidden_grid: vec![],
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
        p.bind_records(
            ResourceId(0),
            vec![rec(0, 7, &[(0, 1, 0), (8, 2, 16)])],
            &SealedTableRegistry::new(),
        )
        .unwrap();
        assert!(p.nodes[0].lowering.is_pm4(), "{:?}", p.nodes[0].lowering);
        assert_eq!(
            p.binding_record_image(ResourceId(0), &p.resources)
                .unwrap()
                .pointers,
            vec![(0, B), (8, C + 16)]
        );
    }
}
