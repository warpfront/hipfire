// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The runtime half of the `SealedTable` conditional certificate (receipt
//! schema 9 / section 3, `required_witnesses = [sealed_table]`).
//!
//! A certified kernel does `i = idx[..]; if ((unsigned)i >= n) return; P =
//! (char*)table[i];` and afterwards only **reads** through `P` (and values
//! derived from `P`). The receipt proves that shape; it is valid only while
//! the runtime holds a [`SealedTableWitness`] for exactly this window:
//!
//! 1. the table argument resolves to a loader-**sealed** immutable region
//!    ([`SealedTableRegistry::seal`]) of at least `n` entries whose live
//!    content still hashes to the sealed digest ([`TableContent`]),
//! 2. every entry points into a registered immutable *pointee* allocation
//!    (checked once, at seal, against the exact table bytes), and
//! 3. nothing in the program writes the table allocation or any pointee
//!    allocation (full-program closure, overlap tests on bindings).
//!
//! The receipt's named assumption — pointee accesses stay inside their
//! pointee allocation — is the certified kernel's; the runtime cannot check
//! it. Nothing here mutates the cached global facts: a node with a sealed
//! table stays [`crate::kernel::Lowering::Conditional`] (a non-PM4 full
//! barrier) until [`crate::window::PreparedWindow::attach_sealed_table_witness`]
//! consumes the witness this module issues, and `submit` re-verifies it.
//!
//! Pointees are read-only and immutable: they add **no** hazard edge and no
//! effect to the program (they are not program resources at all); the bind
//! instead refuses any program that could write them.
//!
//! What this module cannot see is a writer **outside** the program (another
//! queue, a host mapping). The owner must not enqueue one while a table is
//! sealed; [`SealedTableRegistry::retire`] is refused while a window using
//! the table is in flight, and a retire advances the generation so every
//! earlier witness is stale.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

use sha2::{Digest, Sha256};

use crate::kernel::{
    ArenaSealedBinding, Arg, KernelProgram, Lowering, TableBound, WITNESS_SEALED_TABLE,
};
use crate::window::{AllocationIdentity, Quiescence, WindowStep};
use crate::{Binding, ResourceId};

/// Bytes of one table entry (a 64-bit pointer).
pub const ENTRY_BYTES: u64 = 8;

/// The largest bound a sealed table's guard may carry. The certificate's
/// entry-index analysis accepts a sign-extended 32-bit index under an
/// *unsigned* guard `index < bound`, which only bounds the 64-bit address
/// when `bound <= i32::MAX`; a larger bound (even within the sealed entry
/// count) is refused by every bind.
pub const MAX_BOUND: u32 = i32::MAX as u32;

/// Whether [`SealedTableRegistry::seal_with`] accepts null table entries. The
/// owner passes `Allow` only for a table whose certificate (the receipt's
/// `SealedTableRecord::null_guarded`) proves the kernel reads through an entry
/// only when it is nonzero; every other table must hold none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullEntries {
    Refuse,
    Allow,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SealedTableError {
    #[error("seal refused: {0}")]
    Seal(String),
    #[error("table not sealed: {0}")]
    NotSealed(String),
    #[error("wrong allocation: {0}")]
    WrongAllocation(String),
    #[error("node {node}: {detail}")]
    Node { node: usize, detail: String },
    #[error("node {node} table +{table_arg}: bound {bound} exceeds the largest sealed index bound {MAX_BOUND} (a sign-extended index is only sound below 2^31)")]
    BoundTooLarge {
        node: usize,
        table_arg: u32,
        bound: u32,
    },
    #[error("node {node} table +{table_arg}: bound {bound} exceeds the {entries} sealed entries")]
    BoundExceedsEntries {
        node: usize,
        table_arg: u32,
        bound: u32,
        entries: u32,
    },
    #[error("table content: {0}")]
    Content(String),
    #[error("sealed table {base:#x} was modified after sealing: the live digest differs")]
    Modified { base: u64 },
    #[error("stale or foreign witness: {0}")]
    Stale(String),
    #[error("closure: {0}")]
    Closure(String),
    #[error("retire refused: {0}")]
    Retire(String),
    #[error("node {node} table +{table_arg}: the sealed table holds null entries but the certificate records no null guard")]
    NullEntryUnguarded { node: usize, table_arg: u32 },
}

/// An immutable allocation a sealed table's entries point into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pointee {
    pub allocation: Binding,
    /// The owner's assertion that nothing writes the allocation while the
    /// table is sealed (loaded weights). [`SealedTableRegistry::seal`] refuses
    /// a pointee that is not.
    pub immutable: bool,
}

impl Pointee {
    pub fn immutable(allocation: Binding) -> Self {
        Self {
            allocation,
            immutable: true,
        }
    }
}

/// A sealed pointer table: the identity every witness binds to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedTable {
    /// The whole allocation holding the table; a program resource must equal
    /// it exactly (base and bytes).
    pub allocation: Binding,
    pub entries: u32,
    /// Bytes per entry: [`ENTRY_BYTES`].
    pub entry_width: u32,
    /// SHA-256 of the `entries * 8` sealed table bytes.
    pub content_sha256: [u8; 32],
    /// The registered pointee allocations, sorted by base, pairwise disjoint.
    pub pointees: Vec<Pointee>,
    /// The generation the owner sealed at (the allocation's registry
    /// generation: a program's [`AllocationIdentity::generation`] must equal
    /// it).
    pub generation: u64,
    /// Some entry is null (only possible when sealed with
    /// [`NullEntries::Allow`]): a bind needs a certificate with a null guard.
    pub has_null_entries: bool,
}

/// The owner's view of the live table bytes: downloads (or reads) `entries`
/// 8-byte entries of the allocation. The bind compares their digest with the
/// sealed one, so a table modified after sealing is refused. Keeps this crate
/// GPU-free.
pub trait TableContent {
    fn read_table(&self, allocation: Binding, entries: usize) -> Result<Vec<u8>, String>;
}

struct Sealed {
    table: SealedTable,
    /// Windows using the table that are in flight.
    leases: u32,
    /// Unknown quiescence after a window: the table can never be retired.
    quarantined: bool,
}

#[derive(Default)]
struct Inner {
    /// Sealed tables by allocation base.
    tables: BTreeMap<u64, Sealed>,
    /// The lowest generation a new seal at `base` may use (a retire raises it
    /// past the retired seal's).
    floor: BTreeMap<u64, u64>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn short(d: &[u8; 32]) -> String {
    d[..6].iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn overlaps(a: Binding, b: Binding) -> bool {
    a.base < b.base.saturating_add(b.bytes) && b.base < a.base.saturating_add(a.bytes)
}

fn le32(image: &[u8], at: u32) -> Option<u32> {
    let at = at as usize;
    Some(u32::from_le_bytes(image.get(at..at + 4)?.try_into().ok()?))
}

fn le64(image: &[u8], at: u32) -> Option<u64> {
    let at = at as usize;
    Some(u64::from_le_bytes(image.get(at..at + 8)?.try_into().ok()?))
}

/// The registry of sealed tables, filled by the owner/loader at load time and
/// shared (cheaply cloned) with every window that binds against it.
#[derive(Clone, Default)]
pub struct SealedTableRegistry {
    inner: Arc<Mutex<Inner>>,
}

impl std::fmt::Debug for SealedTableRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = lock(&self.inner);
        f.debug_struct("SealedTableRegistry")
            .field("tables", &s.tables.len())
            .finish()
    }
}

impl SealedTableRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        lock(&self.inner).tables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The table sealed at allocation `base`, if any.
    pub fn table_at(&self, base: u64) -> Option<SealedTable> {
        lock(&self.inner).tables.get(&base).map(|s| s.table.clone())
    }

    /// [`Self::seal_with`] for a table whose entries must all be non-null.
    /// Seal a pointer table at load time. The owner passes the exact bytes it
    /// uploaded to the table allocation and the immutable allocations the
    /// entries point into; `seal` **verifies** instead of trusting:
    ///
    /// * `table_bytes.len() == entries * 8` and the allocation holds them;
    /// * the pointees are non-empty, immutable, pairwise disjoint and disjoint
    ///   from the table allocation;
    /// * every entry (little-endian `u64`) is non-null and lies inside exactly
    ///   one pointee allocation;
    /// * no seal exists at, or overlaps, the table allocation, and
    ///   `generation` is not below the one a retired seal raised.
    ///
    /// It records the SHA-256 of the table bytes; the bind later compares the
    /// live table's digest with it.
    ///
    /// # Example: a model owner sealing a table of per-expert weight pointers
    ///
    /// ```
    /// use railgun::sealed_table::{Pointee, SealedTableRegistry};
    /// use railgun::Binding;
    ///
    /// // Two immutable weight allocations and a 3-entry pointer table that
    /// // the loader uploaded into its own allocation.
    /// let weights_a = Binding { base: 0x7000_0000, bytes: 0x10_0000 };
    /// let weights_b = Binding { base: 0x7100_0000, bytes: 0x10_0000 };
    /// let table = Binding { base: 0x7200_0000, bytes: 0x1000 };
    /// let entries = [weights_a.base, weights_b.base + 0x4000, weights_a.base + 0x8000];
    /// let table_bytes: Vec<u8> = entries.iter().flat_map(|p| p.to_le_bytes()).collect();
    ///
    /// let registry = SealedTableRegistry::new();
    /// let sealed = registry
    ///     .seal(
    ///         table,
    ///         entries.len() as u32,
    ///         &table_bytes,
    ///         &[Pointee::immutable(weights_a), Pointee::immutable(weights_b)],
    ///         /* allocation generation */ 1,
    ///     )
    ///     .expect("every entry lies in a registered pointee");
    /// assert_eq!(sealed.entries, 3);
    /// // An entry outside every pointee is refused:
    /// let mut bad = table_bytes.clone();
    /// bad[8..16].copy_from_slice(&0x9000_0000u64.to_le_bytes());
    /// assert!(SealedTableRegistry::new()
    ///     .seal(table, 3, &bad, &[Pointee::immutable(weights_a), Pointee::immutable(weights_b)], 1)
    ///     .is_err());
    /// ```
    pub fn seal(
        &self,
        allocation: Binding,
        entries: u32,
        table_bytes: &[u8],
        pointees: &[Pointee],
        generation: u64,
    ) -> Result<SealedTable, SealedTableError> {
        self.seal_with(
            allocation,
            entries,
            table_bytes,
            pointees,
            generation,
            NullEntries::Refuse,
        )
    }

    /// Seal with an explicit null-entry policy: `Allow` lets entries be null
    /// (every other entry is still checked against the pointees), records
    /// that in [`SealedTable::has_null_entries`], and the bind then admits only
    /// certificates whose table is `null_guarded`.
    pub fn seal_with(
        &self,
        allocation: Binding,
        entries: u32,
        table_bytes: &[u8],
        pointees: &[Pointee],
        generation: u64,
        null_entries: NullEntries,
    ) -> Result<SealedTable, SealedTableError> {
        let refuse = |m: String| Err(SealedTableError::Seal(m));
        if entries == 0 {
            return refuse("a table needs at least one entry".into());
        }
        let need = u64::from(entries) * ENTRY_BYTES;
        if allocation.base == 0 || allocation.base % ENTRY_BYTES != 0 {
            return refuse(format!(
                "table allocation base {:#x} is null or not 8-byte aligned",
                allocation.base
            ));
        }
        if allocation.base.checked_add(allocation.bytes).is_none() {
            return refuse("table allocation base + bytes overflows".into());
        }
        if allocation.bytes < need {
            return refuse(format!(
                "table allocation {:#x}+{} is smaller than {entries} entries ({need} bytes)",
                allocation.base, allocation.bytes
            ));
        }
        if table_bytes.len() as u64 != need {
            return refuse(format!(
                "table bytes are {} long, {entries} entries are {need} bytes",
                table_bytes.len()
            ));
        }
        if pointees.is_empty() {
            return refuse("no pointee allocation is registered".into());
        }
        let mut sorted: Vec<Pointee> = Vec::with_capacity(pointees.len());
        for (i, p) in pointees.iter().enumerate() {
            let b = p.allocation;
            if !p.immutable {
                return refuse(format!(
                    "pointee {i} {:#x}+{} is not immutable",
                    b.base, b.bytes
                ));
            }
            if b.base == 0 || b.bytes == 0 || b.base.checked_add(b.bytes).is_none() {
                return refuse(format!(
                    "pointee {i} {:#x}+{} is null, empty or overflows",
                    b.base, b.bytes
                ));
            }
            if overlaps(b, allocation) {
                return refuse(format!(
                    "pointee {i} {:#x}+{} overlaps the table allocation {:#x}+{}",
                    b.base, b.bytes, allocation.base, allocation.bytes
                ));
            }
            sorted.push(*p);
        }
        sorted.sort_by_key(|p| p.allocation.base);
        if let Some(w) = sorted
            .windows(2)
            .find(|w| overlaps(w[0].allocation, w[1].allocation))
        {
            return refuse(format!(
                "pointee allocations {:#x}+{} and {:#x}+{} overlap",
                w[0].allocation.base,
                w[0].allocation.bytes,
                w[1].allocation.base,
                w[1].allocation.bytes
            ));
        }
        let mut has_null = false;
        for (i, chunk) in table_bytes.chunks_exact(ENTRY_BYTES as usize).enumerate() {
            let ptr = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
            if ptr == 0 {
                if null_entries == NullEntries::Allow {
                    has_null = true;
                    continue;
                }
                return refuse(format!("entry {i} is null"));
            }
            let at = sorted.partition_point(|p| p.allocation.base <= ptr);
            let inside = at
                .checked_sub(1)
                .map(|j| sorted[j].allocation)
                .is_some_and(|b| ptr - b.base < b.bytes);
            if !inside {
                return refuse(format!(
                    "entry {i} = {ptr:#x} lies outside every registered pointee allocation"
                ));
            }
        }

        let mut s = lock(&self.inner);
        if s.tables.contains_key(&allocation.base) {
            return refuse(format!(
                "a table is already sealed at {:#x}",
                allocation.base
            ));
        }
        if let Some(other) = s
            .tables
            .values()
            .find(|o| overlaps(o.table.allocation, allocation))
        {
            return refuse(format!(
                "the allocation overlaps the sealed table at {:#x}+{}",
                other.table.allocation.base, other.table.allocation.bytes
            ));
        }
        let floor = s.floor.get(&allocation.base).copied().unwrap_or(0);
        if generation < floor {
            return refuse(format!(
                "generation {generation} is below {floor}, which the retired seal at {:#x} raised",
                allocation.base
            ));
        }
        let table = SealedTable {
            allocation,
            entries,
            entry_width: ENTRY_BYTES as u32,
            content_sha256: digest(table_bytes),
            pointees: sorted,
            generation,
            has_null_entries: has_null,
        };
        s.tables.insert(
            allocation.base,
            Sealed {
                table: table.clone(),
                leases: 0,
                quarantined: false,
            },
        );
        Ok(table)
    }

    /// Unseal the table at `base`. Every witness issued for it is stale from
    /// now on, and a later seal at `base` must use a higher generation than
    /// the retired one (the returned value is the lowest it may use). Refused
    /// while a window using the table is in flight or quarantined.
    pub fn retire(&self, base: u64) -> Result<u64, SealedTableError> {
        let mut s = lock(&self.inner);
        let Some(entry) = s.tables.get(&base) else {
            return Err(SealedTableError::NotSealed(format!(
                "no sealed table at {base:#x}"
            )));
        };
        if entry.quarantined {
            return Err(SealedTableError::Retire(format!(
                "the table at {base:#x} is quarantined after an unknown-quiescence window"
            )));
        }
        if entry.leases != 0 {
            return Err(SealedTableError::Retire(format!(
                "{} window(s) using the table at {base:#x} are in flight",
                entry.leases
            )));
        }
        let next = entry.table.generation.saturating_add(1);
        s.tables.remove(&base);
        s.floor.insert(base, next);
        Ok(next)
    }

    /// Verify a witness against the live registry without transitioning: every
    /// table it names is still sealed, at the generation, extent, entry count
    /// and digest it was bound for.
    pub fn check_witness(&self, witness: &SealedTableWitness) -> Result<(), SealedTableError> {
        check_in(&lock(&self.inner), witness)
    }

    /// `check_witness`, then mark every table it names as used by a window in
    /// flight (retire is refused until [`TableLease::release`]).
    pub(crate) fn lease(
        &self,
        witness: &SealedTableWitness,
    ) -> Result<TableLease, SealedTableError> {
        let mut s = lock(&self.inner);
        check_in(&s, witness)?;
        let bases: Vec<u64> = witness
            .nodes
            .iter()
            .flat_map(|n| n.tables.iter().map(|t| t.allocation.base))
            .collect();
        for base in &bases {
            s.tables.get_mut(base).expect("checked").leases += 1;
        }
        Ok(TableLease {
            registry: self.clone(),
            bases,
        })
    }

    /// The checked bind (the only source of a [`SealedTableWitness`]): for
    /// every sealed-table node of `input.program`, from the window's bound
    /// state,
    ///
    /// * each table argument is a program resource that equals a sealed
    ///   table's allocation exactly, the bound kernarg pointer is that
    ///   allocation's base, and the allocation generation is the sealed one;
    /// * the guard bound `n` (the bound kernarg scalar of the **encoded
    ///   image**, or the receipt constant) is `<=` the sealed entry count (a
    ///   table whose bound is 0 is vacuous and needs no seal at all);
    /// * the live table content hashes to the sealed digest;
    ///
    /// and for the whole program: no node (kernel effect, atomic, copy or fill
    /// destination) can write the table allocation or any pointee allocation,
    /// and no node has unknown effects (`HipDirect`).
    pub fn bind_sealed_table(
        &self,
        input: &TableBindInputs<'_>,
        content: &dyn TableContent,
    ) -> Result<SealedTableWitness, SealedTableError> {
        let program = input.program;
        let closure = |m: String| Err(SealedTableError::Closure(m));
        if input.bindings.len() != program.resources.len()
            || input.allocations.len() != program.resources.len()
            || input.images.len() != program.nodes.len()
            || input.grids.len() != program.nodes.len()
        {
            return closure("bound state does not cover the program".into());
        }
        let nodes = program.sealed_table_nodes();
        if nodes.is_empty() {
            return closure("the program has no sealed-table node".into());
        }
        let mut verified: BTreeMap<u64, SealedTable> = BTreeMap::new();
        let mut bound_nodes = Vec::with_capacity(nodes.len());
        for index in nodes {
            bound_nodes.push(self.bind_node(input, index, content, &mut verified)?);
        }
        let mut protected: Vec<(Binding, &'static str)> = Vec::new();
        for t in verified.values() {
            protected.push((t.allocation, "sealed table allocation"));
            for p in &t.pointees {
                protected.push((p.allocation, "sealed pointee allocation"));
            }
        }
        check_closure(input, &protected)?;
        let witness = SealedTableWitness {
            window: input.window,
            shape_generation: input.shape_generation,
            nodes: bound_nodes,
        };
        // A retire/re-seal while the live content was being read.
        self.check_witness(&witness)?;
        Ok(witness)
    }

    fn bind_node(
        &self,
        input: &TableBindInputs<'_>,
        index: usize,
        content: &dyn TableContent,
        verified: &mut BTreeMap<u64, SealedTable>,
    ) -> Result<BoundTableNode, SealedTableError> {
        let bad = |d: String| {
            Err(SealedTableError::Node {
                node: index,
                detail: d,
            })
        };
        let node = &input.program.nodes[index];
        let Some(kref) = node.kref.as_ref() else {
            return bad("no certified facts".into());
        };
        let arena_sealed: &[ArenaSealedBinding] =
            node.arena.as_ref().map_or(&[], |a| a.sealed.as_slice());
        // The case (and so the receipt witnesses and digest) this node's
        // sealed tables belong to: its own `table` case, or its bound arena
        // templates'. A node with both is not a shape any receipt names.
        let (witnesses, case_digest) = match (&node.table, &node.arena) {
            (Some(_), Some(a)) if !a.sealed.is_empty() => {
                return bad("carries both a sealed-table case and sealed arena templates".into())
            }
            (Some(case), _) => (&case.witnesses, &case.case_digest),
            (None, Some(a)) if !a.sealed.is_empty() => (&a.witnesses, &a.case_digest),
            _ => return bad("carries no sealed table".into()),
        };
        if !witnesses.iter().any(|w| w == WITNESS_SEALED_TABLE) {
            return bad(
                "receipt does not require the sealed_table witness this bind provides".into(),
            );
        }
        let image = &input.images[index];
        let mut tables = Vec::new();
        let mut vacuous = Vec::new();
        for t in node.table.iter().flat_map(|c| c.tables.iter()) {
            let off = t.table_arg_offset;
            // The guard bound this launch carries: the encoded image's scalar,
            // or the receipt constant.
            let bound = match t.bound {
                TableBound::Const(c) => c,
                TableBound::Kernarg(at) => {
                    le32(image, at).ok_or_else(|| SealedTableError::Node {
                        node: index,
                        detail: format!("short kernarg image: no bound at +{at}"),
                    })?
                }
            };
            // A zero bound makes the certified guard `index < 0` (unsigned, or
            // the signed pair `0 <= index < 0`) unsatisfiable: no entry load
            // can execute, so the case is vacuous for this launch and needs no
            // seal, allocation equality or content digest (the argument's own
            // read stays an ordinary effect). The bound lives in the encoded
            // image, which `matches_window` re-checks at submit; every nonzero
            // bound is still bound strictly below.
            if bound == 0 {
                vacuous.push(off);
                continue;
            }
            let Some(resource) = node.args.iter().find_map(|(o, a)| match a {
                Arg::Resource { resource, .. } if *o == off => Some(*resource),
                _ => None,
            }) else {
                return bad(format!("table argument +{off} is not a bound resource"));
            };
            let value = le64(image, off).ok_or_else(|| SealedTableError::Node {
                node: index,
                detail: "short kernarg image".into(),
            })?;
            tables.push(self.verify_table(
                input,
                index,
                resource,
                off,
                Some(value),
                bound,
                t.null_guarded,
                content,
                verified,
            )?);
        }
        // The sealed tables of the bound arena templates: the pointer lives
        // in a record railgun wrote from the bindings (the base of the table
        // resource), the bound in a scalar railgun declared; both are
        // immutable during replay. The sealed identity `bind_records`
        // discharged must still be the registry's.
        for a in arena_sealed {
            let t = self.verify_table(
                input,
                index,
                a.resource,
                a.table_arg_offset,
                None,
                a.bound,
                a.null_guarded,
                content,
                verified,
            )?;
            if t.entries != a.entries
                || t.generation != a.generation
                || t.content_sha256 != a.content_sha256
                || t.allocation != a.allocation
            {
                return Err(SealedTableError::Stale(format!(
                    "node {index} template {} table +{}: the registry's seal at {:#x} is not the one bind_records discharged (rebind the records)",
                    a.template, a.table_arg_offset, a.allocation.base
                )));
            }
            tables.push(t);
        }
        Ok(BoundTableNode {
            node: index,
            symbol: node.symbol.clone(),
            artifact_sha256: kref.artifact_sha256.clone(),
            receipt_sha256: kref.receipt_sha256.clone(),
            case_digest: case_digest.clone(),
            tables,
            vacuous,
            kernarg_digest: digest(image),
            grid: input.grids[index],
        })
    }

    /// One nonzero-bound table of node `index`: `resource` (the table
    /// argument's, or an arena record's table field's) is exactly a sealed
    /// table's allocation at the sealed generation, the pointer the kernel
    /// sees is its base (`value`, when it is read from the kernarg image; an
    /// arena record's pointer is `base(resource)` by construction), the
    /// bound is at most the sealed entry count, null entries meet a
    /// null-guarded certificate, and the live content hashes to the sealed
    /// digest.
    #[allow(clippy::too_many_arguments)]
    fn verify_table(
        &self,
        input: &TableBindInputs<'_>,
        index: usize,
        resource: ResourceId,
        off: u32,
        value: Option<u64>,
        bound: u32,
        null_guarded: bool,
        content: &dyn TableContent,
        verified: &mut BTreeMap<u64, SealedTable>,
    ) -> Result<BoundTable, SealedTableError> {
        let bad = |d: String| {
            Err(SealedTableError::Node {
                node: index,
                detail: d,
            })
        };
        let (Some(b), Some(alloc)) = (
            input.bindings.get(resource.0 as usize).copied(),
            input.allocations.get(resource.0 as usize).copied(),
        ) else {
            return bad(format!("unknown resource {}", resource.0));
        };
        if alloc.base != b.base || alloc.bytes != b.bytes {
            return bad(format!(
                "resource {} allocation identity differs from its binding",
                resource.0
            ));
        }
        let Some(sealed) = self.table_at(b.base) else {
            return Err(SealedTableError::NotSealed(format!(
                "no sealed table at {:#x}+{} (node {index} table argument +{off})",
                b.base, b.bytes
            )));
        };
        if sealed.allocation != b {
            return Err(SealedTableError::WrongAllocation(format!(
                "node {index} table argument +{off} is resource {} {:#x}+{}, the table sealed at that base is {:#x}+{}",
                resource.0, b.base, b.bytes, sealed.allocation.base, sealed.allocation.bytes
            )));
        }
        if let Some(value) = value {
            if value != b.base {
                return Err(SealedTableError::WrongAllocation(format!(
                    "node {index} table argument +{off} = {value:#x} is not the sealed allocation base {:#x}",
                    b.base
                )));
            }
        }
        if alloc.generation != sealed.generation {
            return Err(SealedTableError::Stale(format!(
                "table at {:#x} was sealed at generation {}, the bound allocation is at {}",
                b.base, sealed.generation, alloc.generation
            )));
        }
        if sealed.has_null_entries && !null_guarded {
            return Err(SealedTableError::NullEntryUnguarded {
                node: index,
                table_arg: off,
            });
        }
        if bound > MAX_BOUND {
            return Err(SealedTableError::BoundTooLarge {
                node: index,
                table_arg: off,
                bound,
            });
        }
        if bound > sealed.entries {
            return Err(SealedTableError::BoundExceedsEntries {
                node: index,
                table_arg: off,
                bound,
                entries: sealed.entries,
            });
        }
        if verified.get(&b.base) != Some(&sealed) {
            let live = content
                .read_table(sealed.allocation, sealed.entries as usize)
                .map_err(SealedTableError::Content)?;
            if live.len() as u64 != u64::from(sealed.entries) * ENTRY_BYTES {
                return Err(SealedTableError::Content(format!(
                    "the owner returned {} bytes for {} entries",
                    live.len(),
                    sealed.entries
                )));
            }
            if digest(&live) != sealed.content_sha256 {
                return Err(SealedTableError::Modified { base: b.base });
            }
            verified.insert(b.base, sealed.clone());
        }
        Ok(BoundTable {
            table_arg_offset: off,
            allocation: sealed.allocation,
            entries: sealed.entries,
            bound,
            generation: sealed.generation,
            content_sha256: sealed.content_sha256,
        })
    }
}

fn check_in(s: &Inner, witness: &SealedTableWitness) -> Result<(), SealedTableError> {
    let stale = |m: String| Err(SealedTableError::Stale(m));
    for n in &witness.nodes {
        for t in &n.tables {
            let base = t.allocation.base;
            let Some(cur) = s.tables.get(&base) else {
                return stale(format!("the table at {base:#x} was retired"));
            };
            if cur.table.generation != t.generation {
                return stale(format!(
                    "table at {base:#x}: witness generation {} is not the sealed {}",
                    t.generation, cur.table.generation
                ));
            }
            if cur.table.allocation != t.allocation
                || cur.table.entries != t.entries
                || cur.table.content_sha256 != t.content_sha256
            {
                return stale(format!(
                    "table at {base:#x} is another seal (digest {} vs witness {})",
                    short(&cur.table.content_sha256),
                    short(&t.content_sha256)
                ));
            }
        }
    }
    Ok(())
}

/// A window using sealed tables is in flight: retire is refused until it is
/// released. Dropping a lease without releasing it (a submission never
/// completed) keeps the tables leased, like an in-flight window.
#[derive(Debug)]
pub(crate) struct TableLease {
    registry: SealedTableRegistry,
    bases: Vec<u64>,
}

impl TableLease {
    /// The window finished (or its submit failed). Proven quiescence frees the
    /// tables; unknown quiescence quarantines them (never retirable).
    pub(crate) fn release(self, quiescence: Quiescence) {
        let mut s = lock(&self.registry.inner);
        for base in &self.bases {
            if let Some(entry) = s.tables.get_mut(base) {
                match quiescence {
                    Quiescence::Proven => entry.leases = entry.leases.saturating_sub(1),
                    Quiescence::Unknown => entry.quarantined = true,
                }
            }
        }
    }
}

/// The program's bound state, as the sealed-table bind sees it (the same
/// shape the indexed bind reads).
pub struct TableBindInputs<'a> {
    pub window: u64,
    pub shape_generation: u64,
    pub program: &'a KernelProgram,
    pub steps: &'a [WindowStep],
    /// One binding per program resource.
    pub bindings: &'a [Binding],
    /// One allocation identity per program resource.
    pub allocations: &'a [AllocationIdentity],
    /// The encoded kernarg image of every node.
    pub images: &'a [Vec<u8>],
    pub grids: &'a [[u32; 3]],
}

/// The no-writer closure: no node of the program, and no copy/fill step, can
/// write a protected allocation, and no node has unknown effects.
fn check_closure(
    input: &TableBindInputs<'_>,
    protected: &[(Binding, &'static str)],
) -> Result<(), SealedTableError> {
    let closure = |m: String| Err(SealedTableError::Closure(m));
    let program = input.program;
    let hit = |b: Binding| protected.iter().find(|(p, _)| overlaps(*p, b));
    for (index, node) in program.nodes.iter().enumerate() {
        if node.kref.is_none() || matches!(node.lowering, Lowering::HipDirect { .. }) {
            return closure(format!(
                "node {index} ({}) has unknown effects (no certified facts or HipDirect): the no-writer proof is impossible",
                node.symbol
            ));
        }
        for e in node.effects.iter().filter(|e| e.writes()) {
            let b = input.bindings.get(e.resource.0 as usize).ok_or_else(|| {
                SealedTableError::Closure(format!(
                    "node {index}: unknown resource {}",
                    e.resource.0
                ))
            })?;
            if let Some((p, what)) = hit(*b) {
                return closure(format!(
                    "node {index} ({}) writes the {what} {:#x}+{} (+{} on resource {})",
                    node.symbol, p.base, p.bytes, e.arg_offset, e.resource.0
                ));
            }
        }
    }
    for (i, step) in input.steps.iter().enumerate() {
        let dst = match step {
            WindowStep::Kernel { .. } => continue,
            WindowStep::Copy(c) => c.dst.resource,
            WindowStep::Fill(f) => f.dst.resource,
        };
        let b = input.bindings.get(dst.0 as usize).ok_or_else(|| {
            SealedTableError::Closure(format!("step {i}: unknown resource {}", dst.0))
        })?;
        if let Some((p, what)) = hit(*b) {
            return closure(format!(
                "step {i} writes the {what} {:#x}+{} (resource {})",
                p.base, p.bytes, dst.0
            ));
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ witness

/// One table of a bound node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundTable {
    pub table_arg_offset: u32,
    pub allocation: Binding,
    pub entries: u32,
    /// The guard bound the bound kernarg image carries.
    pub bound: u32,
    pub generation: u64,
    pub content_sha256: [u8; 32],
}

/// The per-node facts a bind discharged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundTableNode {
    pub node: usize,
    pub symbol: String,
    pub artifact_sha256: String,
    pub receipt_sha256: String,
    pub case_digest: String,
    pub tables: Vec<BoundTable>,
    /// Table argument offsets whose recorded bound is 0 in this launch: the
    /// certified guard is unsatisfiable, so the case is vacuous and needs no
    /// seal (the node is still discharged under its case digest).
    pub vacuous: Vec<u32>,
    pub kernarg_digest: [u8; 32],
    pub grid: [u32; 3],
}

/// Runtime-owned bind witness of the `sealed_table` kind. Private fields, no
/// public constructor: only [`SealedTableRegistry::bind_sealed_table`]
/// produces one, [`crate::window::PreparedWindow::attach_sealed_table_witness`]
/// holds it, and `submit` re-verifies it.
#[derive(Debug)]
pub struct SealedTableWitness {
    window: u64,
    shape_generation: u64,
    nodes: Vec<BoundTableNode>,
}

impl SealedTableWitness {
    pub fn window(&self) -> u64 {
        self.window
    }

    pub fn shape_generation(&self) -> u64 {
        self.shape_generation
    }

    pub fn nodes(&self) -> &[BoundTableNode] {
        &self.nodes
    }

    /// `(program node index, receipt case digest, witness kind)` of every
    /// guarded case this bind discharged: the source of the
    /// `ConditionalWitness` a window manifest admission accepts.
    pub fn discharged_cases(&self) -> Vec<(usize, &str, &'static str)> {
        self.nodes
            .iter()
            .map(|n| (n.node, n.case_digest.as_str(), WITNESS_SEALED_TABLE))
            .collect()
    }

    /// Every bound node's kernarg image and grid still equal the bound bytes
    /// (any post-bind kernarg/shape change is refused at submit).
    pub(crate) fn matches_window(
        &self,
        window: u64,
        shape_generation: u64,
        images: &[Vec<u8>],
        grids: &[[u32; 3]],
    ) -> Result<(), SealedTableError> {
        if self.window != window {
            return Err(SealedTableError::Stale(
                "witness is another window's".into(),
            ));
        }
        if self.shape_generation != shape_generation {
            return Err(SealedTableError::Stale(format!(
                "witness shape generation {} is not the bound {shape_generation}",
                self.shape_generation
            )));
        }
        for n in &self.nodes {
            let image = images.get(n.node).ok_or_else(|| SealedTableError::Node {
                node: n.node,
                detail: "no kernarg image".into(),
            })?;
            if digest(image) != n.kernarg_digest {
                return Err(SealedTableError::Node {
                    node: n.node,
                    detail: "kernarg image changed after bind".into(),
                });
            }
            if grids.get(n.node) != Some(&n.grid) {
                return Err(SealedTableError::Node {
                    node: n.node,
                    detail: "grid changed after bind".into(),
                });
            }
        }
        Ok(())
    }
}

pub mod generation;
pub use generation::{
    acquire_window, acquire_window_owned, finish_window, poll_window, return_to_kv, BlockReclaimed,
    BlockRegistry, BlockUse, Evicting, Expert, FenceBackend, FenceFailure, FenceState,
    FencedWindow, FillComplete, FillTicket, Filling, FillsNotDone, Free, GenerationClosure,
    GenerationId, GenerationRetired, Kv, LentBlock, OwnedSlot, OwnedTableGeneration,
    PeerReaderFence, PendingFills, PendingOrQuarantined, PendingRestore, PendingWindow,
    QuarantinedWindow, Quiesced, ResidencyError, Resident, ScratchLoan, SlotBank, SlotCell,
    SlotOrTable, SlotPhaseView, SlotRead, SlotReads, SlotSpan, Staging, TableBacking,
    WindowReadLease,
};

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod arena;
