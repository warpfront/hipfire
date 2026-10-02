//! railgun: the certified dispatch program (railgun design §1.1). This is the
//! milestone MC slice of the program model: what copies need and nothing
//! else — host words, affine formulas, resources including `PointerTable`,
//! `Node::Copy` and `Node::CopyBatch` — plus [`Program::prepare`], which
//! binds a program to allocations and one assignment of its words, checks
//! it, writes each pointer table's image, and derives every node's region
//! effects. M1's kernel nodes, hazard edges, the gfx12 cache table, the PM4
//! boundary plan and the G7 shadow diff are in [`kernel`], [`plan`] and
//! [`shadow`]; transactions and the graph lowering are later milestones.
//!
//! Copy semantics are `hipMemcpyDtoD`'s. Nodes run in program order; the
//! entries of one `CopyBatch` run concurrently, so `prepare` requires them to
//! be order-independent (pairwise-disjoint destinations, no destination
//! overlapping another entry's source), which makes the batch byte-identical
//! to its entries' memcpys issued in table order.
//!
//! A `PointerTable` is written by railgun at prepare from the program's own
//! bindings and is never a copy destination, so it is immutable during
//! replay. It holds no pointers: each entry names a source and a
//! destination *slot* (a pointer argument of the batch launch, bound to one
//! resource) and 32-bit byte offsets into them ([`copy::CopyEntry`]). A1
//! attributes every access of `railgun_copy_batch` to its slot arguments
//! (see [`copy`]); where inside each resource the accesses land is the
//! table's content, which `prepare` bounds-checks entry by entry
//! ([`Via::Slot`]).
pub mod copy;
pub mod kernel;
pub mod plan;
pub mod shadow;

use std::ops::RangeInclusive;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WordId(pub u32);

/// A host word (§1.1 `Word`, `source: Host`): patched at submit, refused
/// outside its domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Word {
    pub name: String,
    pub domain: RangeInclusive<u64>,
}

/// `c0 + Σ ci·wi` over host words (§1.1 `Formula`; `ceil_div` is not
/// needed by copies). Coefficients are unsigned: copy offsets and lengths
/// are non-negative combinations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Formula {
    pub c0: u64,
    pub terms: Vec<(WordId, u64)>,
}

impl Formula {
    pub fn constant(c0: u64) -> Self {
        Self { c0, terms: Vec::new() }
    }

    /// `coeff · word`.
    pub fn word(word: WordId, coeff: u64) -> Self {
        Self { c0: 0, terms: vec![(word, coeff)] }
    }

    pub fn plus(mut self, c: u64) -> Self {
        self.c0 += c;
        self
    }

    /// Value under `words` (indexed by `WordId`), with overflow refused.
    pub fn eval(&self, words: &[u64]) -> Result<u64, PrepareError> {
        self.terms.iter().try_fold(self.c0, |acc, &(w, c)| {
            let value = *words.get(w.0 as usize).ok_or(PrepareError::UnknownWord(w.0))?;
            c.checked_mul(value).and_then(|t| acc.checked_add(t)).ok_or(PrepareError::Overflow)
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId(pub u32);

/// §1.1 `Resource.kind`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    Weights,
    Kv,
    DnState,
    ConvState,
    EfResidual,
    Tape,
    HiddenRing,
    DraftScratch,
    VerifyScratch,
    Scratch,
    Logits,
    ControlWord,
    /// Copy descriptors, one [`copy::CopyEntry`] per [`CopySpec`], written
    /// at prepare and read by `railgun_copy_batch`. Entries' resources are
    /// static (only offsets and lengths depend on words), so each table has
    /// one slot map for every assignment.
    PointerTable(Vec<CopySpec>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resource {
    pub label: String,
    /// Bytes the program may address, from the binding's base.
    pub size: Formula,
    pub kind: ResourceKind,
}

/// A resource's allocation for one prepare (§1.1 `binding` without the
/// revision and VMM reservation, which arrive with M1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Binding {
    pub base: u64,
    pub bytes: u64,
}

/// A byte address inside a resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Addr {
    pub resource: ResourceId,
    pub offset: Formula,
}

/// `len` bytes from `src` to `dst`: the payload of `Node::Copy` and one
/// pointer-table entry (§1.1 `Copy { src, dst, len }`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopySpec {
    pub src: Addr,
    pub dst: Addr,
    pub len: Formula,
}

/// §1.1 `Node`, copy variants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    /// One copy: `railgun_copy` in the PM4 lowering, a memcpy node in the
    /// graph lowering.
    Copy(CopySpec),
    /// The first `n` entries of a `PointerTable` resource, one
    /// `railgun_copy_batch` launch.
    CopyBatch { table: ResourceId, n: Formula },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub arch: String,
    pub words: Vec<Word>,
    pub resources: Vec<Resource>,
    pub nodes: Vec<Node>,
}

/// A concrete byte range of a resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub resource: ResourceId,
    pub offset: u64,
    pub len: u64,
}

impl Span {
    pub fn end(&self) -> u64 {
        self.offset + self.len
    }

    pub fn overlaps(&self, other: &Span) -> bool {
        self.resource == other.resource && self.len > 0 && other.len > 0 && self.offset < other.end() && other.offset < self.end()
    }
}

/// One copy under a word assignment: spans and device addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedCopy {
    pub src: Span,
    pub dst: Span,
    pub src_addr: u64,
    pub dst_addr: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
}

/// What attributes an effect's address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Via {
    /// A kernarg pointer (A1 attributes the access to it).
    Kernarg,
    /// Slot `slot` of a `CopyBatch` over table `table`: A1 attributes the
    /// access to that slot's pointer argument (source slots are read,
    /// destination slots written), which is bound to the effect's resource;
    /// the span inside it is an entry of the table, which railgun wrote and
    /// bounds-checked at prepare.
    Slot { table: ResourceId, slot: u32 },
}

/// §1.1 `RegionEffect`, concrete for one word assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Effect {
    pub span: Span,
    pub access: Access,
    pub via: Via,
}

/// A `CopyBatch` slot argument: the resource bound to it and its base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub resource: ResourceId,
    pub base: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreparedNode {
    Copy(ResolvedCopy),
    /// `src_slots[k]` is bound to source argument `s<k>` and `dst_slots[k]`
    /// to destination argument `d<k>`.
    CopyBatch { table: ResourceId, table_addr: u64, src_slots: Vec<Slot>, dst_slots: Vec<Slot>, copies: Vec<ResolvedCopy> },
}

impl PreparedNode {
    pub fn effects(&self) -> Vec<Effect> {
        match self {
            Self::Copy(c) if c.dst.len == 0 => Vec::new(),
            Self::Copy(c) => vec![
                Effect { span: c.src, access: Access::Read, via: Via::Kernarg },
                Effect { span: c.dst, access: Access::Write, via: Via::Kernarg },
            ],
            Self::CopyBatch { table, src_slots, dst_slots, copies, .. } => {
                let via = |slots: &[Slot], span: &Span| Via::Slot {
                    table: *table,
                    slot: slots.iter().position(|s| s.resource == span.resource).expect("prepare bound every entry's resource") as u32,
                };
                let image = Span { resource: *table, offset: 0, len: copies.len() as u64 * copy::ENTRY_BYTES };
                let mut out = vec![Effect { span: image, access: Access::Read, via: Via::Kernarg }];
                for c in copies.iter().filter(|c| c.dst.len > 0) {
                    out.push(Effect { span: c.src, access: Access::Read, via: via(src_slots, &c.src) });
                    out.push(Effect { span: c.dst, access: Access::Write, via: via(dst_slots, &c.dst) });
                }
                out
            }
        }
    }

    /// The copy-kernel launch, or `None` when there is nothing to copy.
    pub fn launch(&self) -> Option<copy::Launch> {
        match self {
            Self::Copy(c) if c.dst.len == 0 => None,
            Self::Copy(c) => Some(copy::copy_launch(c.src_addr, c.dst_addr, c.dst.len as u32)),
            Self::CopyBatch { copies, .. } if copies.is_empty() => None,
            Self::CopyBatch { table_addr, src_slots, dst_slots, copies, .. } => {
                let max = copies.iter().map(|c| c.dst.len).max().unwrap_or(0);
                let bases = |slots: &[Slot]| slots.iter().map(|s| s.base).collect::<Vec<_>>();
                Some(copy::batch_launch(*table_addr, copies.len() as u32, &bases(src_slots), &bases(dst_slots), max))
            }
        }
    }
}

/// A pointer table's contents under one assignment, to be written to
/// `addr` before the program runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableImage {
    pub resource: ResourceId,
    pub addr: u64,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prepared {
    pub nodes: Vec<PreparedNode>,
    pub tables: Vec<TableImage>,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PrepareError {
    #[error("{got} word values for {want} words")]
    WordCount { got: usize, want: usize },
    #[error("word {name} = {value} outside {lo}..={hi}")]
    WordOutOfDomain { name: String, value: u64, lo: u64, hi: u64 },
    #[error("{got} bindings for {want} resources")]
    BindingCount { got: usize, want: usize },
    #[error("formula references unknown word {0}")]
    UnknownWord(u32),
    #[error("reference to unknown resource {0}")]
    UnknownResource(u32),
    #[error("formula overflows u64")]
    Overflow,
    #[error("resource {label}: binding holds {bound} bytes, size is {size}")]
    BindingTooSmall { label: String, bound: u64, size: u64 },
    #[error("node {node}: [{offset}, {offset}+{len}) is outside resource {label} ({size} bytes)")]
    OutOfBounds { node: usize, label: String, offset: u64, len: u64, size: u64 },
    #[error("node {node}: a {len}-byte copy exceeds the kernel limit {max}")]
    TooLong { node: usize, len: u64, max: u64 },
    #[error("node {node}: source and destination overlap")]
    SelfOverlap { node: usize },
    #[error("node {node}: writes pointer table {label}")]
    WritesTable { node: usize, label: String },
    #[error("node {node}: resource {label} is not a PointerTable")]
    NotATable { node: usize, label: String },
    #[error("node {node}: pointer table {label} is already used by another CopyBatch")]
    SharedTable { node: usize, label: String },
    #[error("node {node}: n = {n}, the table holds {entries} entries and a batch at most {max}")]
    BatchSize { node: usize, n: u64, entries: usize, max: u64 },
    #[error("node {node}: table {label} needs {need} bytes, its size is {size}")]
    TableTooSmall { node: usize, label: String, need: u64, size: u64 },
    #[error("node {node}: entry {a} writes bytes entry {b} {what}")]
    BatchOverlap { node: usize, a: usize, b: usize, what: &'static str },
    #[error("node {node}: table {label} names {count} {side} resources, a batch binds at most {max}")]
    TooManySlots { node: usize, label: String, side: &'static str, count: usize, max: usize },
    #[error("node {node}: entry {entry} reaches [{offset}, {offset}+{len}) of {label}; a batch addresses the first {max} bytes of each slot")]
    SlotRange { node: usize, entry: usize, label: String, offset: u64, len: u64, max: u64 },
}

impl Program {
    fn resource(&self, id: ResourceId) -> Result<&Resource, PrepareError> {
        self.resources.get(id.0 as usize).ok_or(PrepareError::UnknownResource(id.0))
    }

    /// Bind the program to `bindings` (one per resource) under `words` (one
    /// per word) and check it: words in domain, every region inside its
    /// resource, no copy overlapping itself, no copy writing a pointer table,
    /// batches order-independent, each batch's resources within its slots
    /// and every entry within the slot range. Returns the resolved nodes and
    /// the table images to upload.
    pub fn prepare(&self, words: &[u64], bindings: &[Binding]) -> Result<Prepared, PrepareError> {
        if words.len() != self.words.len() {
            return Err(PrepareError::WordCount { got: words.len(), want: self.words.len() });
        }
        for (w, &value) in self.words.iter().zip(words) {
            if !w.domain.contains(&value) {
                let (lo, hi) = (*w.domain.start(), *w.domain.end());
                return Err(PrepareError::WordOutOfDomain { name: w.name.clone(), value, lo, hi });
            }
        }
        if bindings.len() != self.resources.len() {
            return Err(PrepareError::BindingCount { got: bindings.len(), want: self.resources.len() });
        }
        let mut sizes = Vec::with_capacity(self.resources.len());
        for (r, b) in self.resources.iter().zip(bindings) {
            let size = r.size.eval(words)?;
            if b.bytes < size {
                return Err(PrepareError::BindingTooSmall { label: r.label.clone(), bound: b.bytes, size });
            }
            sizes.push(size);
        }
        let env = Env { program: self, words, bindings, sizes: &sizes };
        let mut nodes = Vec::with_capacity(self.nodes.len());
        let mut tables = Vec::new();
        let mut used = vec![false; self.resources.len()];
        for (node, item) in self.nodes.iter().enumerate() {
            match item {
                Node::Copy(spec) => nodes.push(PreparedNode::Copy(env.resolve(node, spec)?)),
                Node::CopyBatch { table, n } => {
                    let t = self.resource(*table)?;
                    let ResourceKind::PointerTable(entries) = &t.kind else {
                        return Err(PrepareError::NotATable { node, label: t.label.clone() });
                    };
                    if std::mem::replace(&mut used[table.0 as usize], true) {
                        return Err(PrepareError::SharedTable { node, label: t.label.clone() });
                    }
                    let count = n.eval(words)?;
                    if count > entries.len() as u64 || count > copy::MAX_BATCH {
                        return Err(PrepareError::BatchSize { node, n: count, entries: entries.len(), max: copy::MAX_BATCH });
                    }
                    let need = count * copy::ENTRY_BYTES;
                    if need > sizes[table.0 as usize] {
                        return Err(PrepareError::TableTooSmall { node, label: t.label.clone(), need, size: sizes[table.0 as usize] });
                    }
                    let (src_ids, dst_ids) = slot_map(node, &t.label, entries)?;
                    let slots = |ids: &[ResourceId]| -> Result<Vec<Slot>, PrepareError> {
                        ids.iter().map(|&r| self.resource(r).map(|_| Slot { resource: r, base: bindings[r.0 as usize].base })).collect()
                    };
                    let (src_slots, dst_slots) = (slots(&src_ids)?, slots(&dst_ids)?);
                    let copies =
                        entries[..count as usize].iter().map(|spec| env.resolve(node, spec)).collect::<Result<Vec<_>, _>>()?;
                    check_independent(node, &copies)?;
                    let mut bytes = Vec::with_capacity(need as usize);
                    for (entry, c) in copies.iter().enumerate() {
                        let place = |ids: &[ResourceId], span: &Span| -> Result<(u32, u32), PrepareError> {
                            let offset = u32::try_from(span.offset).ok().filter(|_| span.end() <= copy::MAX_SLOT_END);
                            let Some(offset) = offset else {
                                let label = self.resource(span.resource)?.label.clone();
                                let (offset, len, max) = (span.offset, span.len, copy::MAX_SLOT_END);
                                return Err(PrepareError::SlotRange { node, entry, label, offset, len, max });
                            };
                            let slot = ids.iter().position(|&r| r == span.resource).expect("slot_map covers every entry");
                            Ok((slot as u32, offset))
                        };
                        let (src_slot, src_offset) = place(&src_ids, &c.src)?;
                        let (dst_slot, dst_offset) = place(&dst_ids, &c.dst)?;
                        copy::CopyEntry { src_slot, src_offset, dst_slot, dst_offset, bytes: c.dst.len as u32 }.encode(&mut bytes);
                    }
                    let table_addr = bindings[table.0 as usize].base;
                    tables.push(TableImage { resource: *table, addr: table_addr, bytes });
                    nodes.push(PreparedNode::CopyBatch { table: *table, table_addr, src_slots, dst_slots, copies });
                }
            }
        }
        Ok(Prepared { nodes, tables })
    }
}

/// The slot map of a table: the distinct source and destination resources
/// its entries name, each in first-use order (slot `k` is the `k`-th). It
/// covers every entry, not only the executed prefix, so a table's slot
/// arguments do not depend on the words.
fn slot_map(node: usize, label: &str, entries: &[CopySpec]) -> Result<(Vec<ResourceId>, Vec<ResourceId>), PrepareError> {
    let (mut src, mut dst) = (Vec::new(), Vec::new());
    for e in entries {
        if !src.contains(&e.src.resource) {
            src.push(e.src.resource);
        }
        if !dst.contains(&e.dst.resource) {
            dst.push(e.dst.resource);
        }
    }
    for (ids, side, max) in [(&src, "source", copy::SRC_SLOTS), (&dst, "destination", copy::DST_SLOTS)] {
        if ids.len() > max {
            return Err(PrepareError::TooManySlots { node, label: label.to_owned(), side, count: ids.len(), max });
        }
    }
    Ok((src, dst))
}

struct Env<'a> {
    program: &'a Program,
    words: &'a [u64],
    bindings: &'a [Binding],
    sizes: &'a [u64],
}

impl Env<'_> {
    fn span(&self, node: usize, at: &Addr, len: u64) -> Result<(Span, u64), PrepareError> {
        let r = self.program.resource(at.resource)?;
        let offset = at.offset.eval(self.words)?;
        let size = self.sizes[at.resource.0 as usize];
        if offset.checked_add(len).is_none_or(|end| end > size) {
            return Err(PrepareError::OutOfBounds { node, label: r.label.clone(), offset, len, size });
        }
        let addr = self.bindings[at.resource.0 as usize].base.checked_add(offset).ok_or(PrepareError::Overflow)?;
        Ok((Span { resource: at.resource, offset, len }, addr))
    }

    fn resolve(&self, node: usize, spec: &CopySpec) -> Result<ResolvedCopy, PrepareError> {
        let len = spec.len.eval(self.words)?;
        if len > copy::MAX_COPY_BYTES {
            return Err(PrepareError::TooLong { node, len, max: copy::MAX_COPY_BYTES });
        }
        let (src, src_addr) = self.span(node, &spec.src, len)?;
        let (dst, dst_addr) = self.span(node, &spec.dst, len)?;
        let dst_resource = self.program.resource(dst.resource)?;
        if len > 0 && matches!(dst_resource.kind, ResourceKind::PointerTable(_)) {
            return Err(PrepareError::WritesTable { node, label: dst_resource.label.clone() });
        }
        if src.overlaps(&dst) {
            return Err(PrepareError::SelfOverlap { node });
        }
        Ok(ResolvedCopy { src, dst, src_addr, dst_addr })
    }
}

/// Order independence of one batch: destinations pairwise disjoint, and no
/// destination overlapping another entry's source. O(n log n).
fn check_independent(node: usize, copies: &[ResolvedCopy]) -> Result<(), PrepareError> {
    let key = |s: &Span| (s.resource, s.offset);
    let mut dsts: Vec<(Span, usize)> = copies.iter().enumerate().filter(|(_, c)| c.dst.len > 0).map(|(i, c)| (c.dst, i)).collect();
    dsts.sort_by_key(|(s, _)| key(s));
    for pair in dsts.windows(2) {
        if pair[0].0.overlaps(&pair[1].0) {
            return Err(PrepareError::BatchOverlap { node, a: pair[0].1, b: pair[1].1, what: "also writes" });
        }
    }
    // Disjoint and sorted, so each resource's destinations have increasing
    // ends: the first destination ending after a source's start is the only
    // candidate for overlapping it.
    for (b, c) in copies.iter().enumerate().filter(|(_, c)| c.src.len > 0) {
        let s = c.src;
        let at = dsts.partition_point(|(d, _)| (d.resource, d.end()) <= (s.resource, s.offset));
        if let Some(&(d, a)) = dsts.get(at) {
            if a != b && d.overlaps(&s) {
                return Err(PrepareError::BatchOverlap { node, a, b, what: "reads" });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const RING: ResourceId = ResourceId(0);
    const DST: ResourceId = ResourceId(1);
    const TABLE: ResourceId = ResourceId(2);
    const POS: WordId = WordId(0);
    const ROWS: WordId = WordId(1);

    fn at(resource: ResourceId, offset: u64) -> Addr {
        Addr { resource, offset: Formula::constant(offset) }
    }
    fn spec(src: Addr, dst: Addr, len: u64) -> CopySpec {
        CopySpec { src, dst, len: Formula::constant(len) }
    }

    /// Rows `r` of a 64-byte-row ring land at `position + r` of `dst`.
    fn program(entries: Vec<CopySpec>, table_size: u64) -> Program {
        let resources = vec![
            Resource { label: "ring".into(), size: Formula::constant(1024), kind: ResourceKind::HiddenRing },
            Resource { label: "dst".into(), size: Formula::constant(4096), kind: ResourceKind::DraftScratch },
            Resource { label: "table".into(), size: Formula::constant(table_size), kind: ResourceKind::PointerTable(entries) },
        ];
        let words = vec![Word { name: "position".into(), domain: 0..=32 }, Word { name: "rows".into(), domain: 1..=4 }];
        Program { arch: "gfx1201".into(), words, resources, nodes: vec![Node::CopyBatch { table: TABLE, n: Formula::word(ROWS, 1) }] }
    }
    fn rows() -> Vec<CopySpec> {
        (0..4)
            .map(|r| CopySpec {
                src: at(RING, 64 * r),
                dst: Addr { resource: DST, offset: Formula::word(POS, 64).plus(64 * r) },
                len: Formula::constant(64),
            })
            .collect()
    }
    fn bind() -> Vec<Binding> {
        vec![Binding { base: 0x10_0000, bytes: 1024 }, Binding { base: 0x20_0000, bytes: 4096 }, Binding { base: 0x30_0000, bytes: 96 }]
    }

    #[test]
    fn batch_prefix_follows_the_words_and_its_table_image_names_slots_and_offsets() {
        let p = program(rows(), 96).prepare(&[5, 3], &bind()).unwrap();
        let [PreparedNode::CopyBatch { copies, table_addr, src_slots, dst_slots, .. }] = &p.nodes[..] else { panic!("{:?}", p.nodes) };
        assert_eq!((*table_addr, copies.len()), (0x30_0000, 3));
        assert_eq!((&src_slots[..], &dst_slots[..]), (&[Slot { resource: RING, base: 0x10_0000 }][..], &[Slot { resource: DST, base: 0x20_0000 }][..]));
        let image = &p.tables[0].bytes;
        assert_eq!(image.len(), 72);
        let entry = |i: usize| -> Vec<u32> {
            image[i * 24..i * 24 + 24].chunks(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect()
        };
        // Entry 2: ring row 2 -> dst row 5 + 2, as (slot, offset) pairs.
        assert_eq!(entry(2), [0, 128, 0, 7 * 64, 64, 0]);
        let effects = p.nodes[0].effects();
        assert_eq!(effects[0], Effect { span: Span { resource: TABLE, offset: 0, len: 72 }, access: Access::Read, via: Via::Kernarg });
        assert!(effects[1..].iter().all(|e| e.via == Via::Slot { table: TABLE, slot: 0 }));
        assert_eq!(effects.len(), 1 + 2 * 3);
        let launch = p.nodes[0].launch().unwrap();
        let arg = |at: usize| u64::from_le_bytes(launch.kernarg[at..at + 8].try_into().unwrap());
        assert_eq!((arg(0), arg(16), arg(24), arg(80), arg(88)), (0x30_0000, 0x10_0000, 0, 0x20_0000, 0));
    }

    #[test]
    fn slots_cover_the_whole_table_and_are_bounded_in_count_and_range() {
        // Entry 3 reads `dst`: it takes source slot 1 even when only the first
        // entry runs, so the launch's slot arguments do not depend on `rows`.
        let mut e = rows();
        e[3] = spec(at(DST, 2048), at(DST, 3072), 64);
        let p = program(e, 96);
        for rows in [1, 4] {
            let prepared = p.prepare(&[0, rows], &bind()).unwrap();
            let PreparedNode::CopyBatch { src_slots, .. } = &prepared.nodes[0] else { unreachable!() };
            assert_eq!(src_slots.iter().map(|s| s.resource).collect::<Vec<_>>(), [RING, DST]);
        }
        // Nine source resources: refused whatever the prefix.
        let mut resources: Vec<Resource> = (0..9)
            .map(|i| Resource { label: format!("src{i}"), size: Formula::constant(64), kind: ResourceKind::Scratch })
            .collect();
        resources.push(Resource { label: "dst".into(), size: Formula::constant(4096), kind: ResourceKind::Scratch });
        let entries = (0..9).map(|i| spec(at(ResourceId(i), 0), at(ResourceId(9), 64 * u64::from(i)), 64)).collect();
        resources.push(Resource { label: "t".into(), size: Formula::constant(9 * 24), kind: ResourceKind::PointerTable(entries) });
        let wide = Program { arch: "gfx1201".into(), words: Vec::new(), resources, nodes: vec![Node::CopyBatch { table: ResourceId(10), n: Formula::constant(1) }] };
        let bindings: Vec<Binding> = (0..11).map(|i| Binding { base: 0x100_0000 * (i + 1), bytes: 4096 }).collect();
        let err = wide.prepare(&[], &bindings).unwrap_err();
        assert_eq!(err, PrepareError::TooManySlots { node: 0, label: "t".into(), side: "source", count: 9, max: copy::SRC_SLOTS });
        // A 5 GiB destination: a batch reaches its first 4 GiB, a single copy all of it.
        let big = 5u64 << 30;
        let mut p = program(Vec::new(), 24);
        p.resources[1].size = Formula::constant(big);
        let mut b = bind();
        b[1].bytes = big;
        for (offset, ok) in [((1 << 32) - 64, true), ((1 << 32) - 32, false), (1 << 32, false)] {
            let entry = spec(at(RING, 0), at(DST, offset), 64);
            p.resources[2].kind = ResourceKind::PointerTable(vec![entry.clone()]);
            p.nodes = vec![Node::CopyBatch { table: TABLE, n: Formula::constant(1) }];
            match p.prepare(&[0, 1], &b) {
                Ok(_) => assert!(ok, "{offset}"),
                Err(e) => assert_eq!((ok, e), (false, PrepareError::SlotRange { node: 0, entry: 0, label: "dst".into(), offset, len: 64, max: 1 << 32 })),
            }
            p.nodes = vec![Node::Copy(entry)];
            assert!(p.prepare(&[0, 1], &b).is_ok());
        }
    }

    #[test]
    fn out_of_domain_words_and_out_of_bounds_regions_are_refused() {
        let p = program(rows(), 96);
        assert!(matches!(p.prepare(&[33, 3], &bind()), Err(PrepareError::WordOutOfDomain { value: 33, .. })));
        // A domain wider than the resource allows is still stopped at the resource end.
        let mut wide = p.clone();
        wide.words[0].domain = 0..=64;
        assert!(wide.prepare(&[60, 4], &bind()).is_ok(), "rows 60..64 end exactly at 4096");
        assert!(matches!(wide.prepare(&[61, 4], &bind()), Err(PrepareError::OutOfBounds { offset: 4096, len: 64, .. })));
        let mut short = bind();
        short[1].bytes = 4095;
        assert!(matches!(p.prepare(&[0, 1], &short), Err(PrepareError::BindingTooSmall { .. })));
        assert!(matches!(program(rows(), 71).prepare(&[0, 3], &bind()), Err(PrepareError::TableTooSmall { need: 72, .. })));
    }

    #[test]
    fn batch_entries_must_be_order_independent() {
        // Two writers of dst bytes [64, 128).
        let mut e = rows();
        e[3] = spec(at(RING, 512), at(DST, 100), 16);
        let err = program(e, 96).prepare(&[1, 4], &bind()).unwrap_err();
        assert_eq!(err, PrepareError::BatchOverlap { node: 0, a: 0, b: 3, what: "also writes" });
        // Entry 1 reads what entry 0 writes (a chain memcpys would order).
        let e = vec![spec(at(RING, 0), at(DST, 0), 64), spec(at(DST, 32), at(DST, 1024), 64)];
        let err = program(e, 48).prepare(&[0, 2], &bind()).unwrap_err();
        assert_eq!(err, PrepareError::BatchOverlap { node: 0, a: 0, b: 1, what: "reads" });
        // Adjacent, touching ranges are independent; an entry may read its own
        // resource elsewhere.
        let e = vec![spec(at(DST, 0), at(DST, 64), 64), spec(at(DST, 256), at(DST, 128), 64)];
        assert!(program(e, 48).prepare(&[0, 2], &bind()).is_ok());
        // Only the executed prefix is checked.
        let mut e = rows();
        e[3] = spec(at(RING, 0), at(DST, 0), 64);
        assert!(program(e, 96).prepare(&[0, 3], &bind()).is_ok());
    }

    #[test]
    fn copies_may_not_overlap_themselves_or_write_a_pointer_table() {
        let mut p = program(rows(), 96);
        p.nodes = vec![Node::Copy(spec(at(DST, 0), at(DST, 63), 64))];
        assert_eq!(p.prepare(&[0, 1], &bind()).unwrap_err(), PrepareError::SelfOverlap { node: 0 });
        p.nodes = vec![Node::Copy(spec(at(RING, 0), at(TABLE, 0), 24))];
        assert!(matches!(p.prepare(&[0, 1], &bind()), Err(PrepareError::WritesTable { .. })));
        p.nodes = vec![Node::CopyBatch { table: DST, n: Formula::constant(1) }];
        assert!(matches!(p.prepare(&[0, 1], &bind()), Err(PrepareError::NotATable { .. })));
        p.nodes = vec![Node::CopyBatch { table: TABLE, n: Formula::constant(1) }; 2];
        assert!(matches!(p.prepare(&[0, 1], &bind()), Err(PrepareError::SharedTable { node: 1, .. })));
        // A zero-length copy has no effect and no launch.
        p.nodes = vec![Node::Copy(spec(at(DST, 0), at(DST, 0), 0))];
        let prepared = p.prepare(&[0, 1], &bind()).unwrap();
        assert_eq!((prepared.nodes[0].effects().len(), prepared.nodes[0].launch()), (0, None));
    }
}
