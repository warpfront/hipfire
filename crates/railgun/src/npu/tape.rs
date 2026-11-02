// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Retained-dispatch tape identity, program cache, and replay verification.
//!
//! Pure data/hashing helpers: no hardware, no allocation on cache hits.
//!
//! # Hash
//!
//! All identities use FNV-1a 64 ([`fnv1a`]) with offset basis
//! [`FNV_OFFSET`] (`0xcbf29ce484222325`) and prime `0x100000001b3`.
//! All integers in canonical layouts are little-endian.
//!
//! # Binding domains
//!
//! A [`BindingKind`] names a word that changes between dispatches of the same
//! retained program. The word index lives in one of two domains:
//!
//! | kind             | tag | payload      | index refers to            |
//! |------------------|-----|--------------|----------------------------|
//! | `ArgAddress{arg}`| 1   | `arg` u8     | cmd BO word (not txn)      |
//! | `InstsAddress`   | 2   | none         | cmd BO word (not txn)      |
//! | `Seq{run}`       | 3   | `run` u32 LE | txn u32 word               |
//! | `Custom(id)`     | 4   | `id` u16 LE  | txn u32 word               |
//!
//! Only `Seq` and `Custom` bindings mask transaction bytes: the full 4 bytes
//! of the indexed txn word are zeroed in the canonical form, so their current
//! values do not affect identity. Cmd-BO indices never touch txn bytes.
//!
//! **Inventory contract (valid in-range):** every `Seq`/`Custom` index MUST be
//! `< txn.len() / 4` (a complete u32 word inside the txn). Constructors
//! ([`ProgramKey::from_spec`]) never panic and never mask unrelated bytes: an
//! out-of-range index masks nothing but is still hashed in the dynamic list.
//! [`ProgramCache::get_or_insert_with`] rejects such specs with an explicit
//! error (see [`ProgramSpec::validate`]). Cmd-BO indices cannot be range-checked
//! here (the cmd BO size is not part of the spec).
//!
//! # Program layout (canonical bytes hashed by [`ProgramKey`])
//!
//! ```text
//! 'P' | u64 pdi.len() | pdi
//! 'T' | u64 txn.len() | masked txn
//! 'D' | u64 dynamic.len() | repeat { u64 word index | kind tag | kind payload }
//! ```
//!
//! The dynamic list preserves caller order (order matters).
//!
//! # Config layout ([`ConfigKey`])
//!
//! ```text
//! 'C' | pdi          (no length prefix)
//! ```
//!
//! # Tape layout ([`Tape::sequence_hash`])
//!
//! ```text
//! repeat per entry {
//!     u64 program key | u64 config key | u64 binding count
//!     | repeat { kind tag | kind payload }
//! }
//! ```
//!
//! Binding *values* are excluded: only kinds and their order matter. The
//! binding counts delimit entries. An empty tape hashes to [`FNV_OFFSET`].

use std::collections::hash_map::Entry;
use std::collections::HashMap;

/// FNV-1a 64 offset basis.
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64 prime.
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Continue an FNV-1a 64 hash with `bytes`, starting from `seed`.
/// Use [`FNV_OFFSET`] as the seed for a fresh hash.
pub fn fnv1a(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = seed;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// What kind of per-dispatch value a dynamic word carries.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BindingKind {
    /// Address of kernel argument `arg`; index is a cmd BO word. Tag 1, payload `arg` u8.
    ArgAddress { arg: u8 },
    /// Address of the instruction buffer; index is a cmd BO word. Tag 2, no payload.
    InstsAddress,
    /// Sequence value for run `run`; index is a txn u32 word. Tag 3, payload `run` u32 LE.
    Seq { run: u32 },
    /// Custom patched word; index is a txn u32 word. Tag 4, payload id u16 LE.
    Custom(u16),
}

impl BindingKind {
    /// Stable kind tag (1..=4).
    pub const fn tag(self) -> u8 {
        match self {
            BindingKind::ArgAddress { .. } => 1,
            BindingKind::InstsAddress => 2,
            BindingKind::Seq { .. } => 3,
            BindingKind::Custom(_) => 4,
        }
    }

    /// True when the index refers to a txn u32 word (and masks it): `Seq`, `Custom`.
    pub const fn masks_txn_word(self) -> bool {
        matches!(self, BindingKind::Seq { .. } | BindingKind::Custom(_))
    }

    /// Canonical encoding: tag byte followed by the payload. Returns the
    /// buffer and the number of valid leading bytes (1..=5).
    pub fn encode(self) -> ([u8; 5], usize) {
        let mut out = [0u8; 5];
        out[0] = self.tag();
        match self {
            BindingKind::ArgAddress { arg } => {
                out[1] = arg;
                (out, 2)
            }
            BindingKind::InstsAddress => (out, 1),
            BindingKind::Seq { run } => {
                out[1..5].copy_from_slice(&run.to_le_bytes());
                (out, 5)
            }
            BindingKind::Custom(id) => {
                out[1..3].copy_from_slice(&id.to_le_bytes());
                (out, 3)
            }
        }
    }
}

/// Borrowed description of a retained program.
///
/// `dynamic` lists `(word index, kind)` in canonical (caller) order. See the
/// module docs for index domains and the in-range inventory contract.
#[derive(Clone, Copy, Debug)]
pub struct ProgramSpec<'a> {
    /// Program image bytes.
    pub pdi: &'a [u8],
    /// Transaction bytes (current values; dynamic words are masked in identity).
    pub txn: &'a [u8],
    /// Dynamic word inventory.
    pub dynamic: &'a [(usize, BindingKind)],
}

/// Identity of a program: hash of the canonical program layout.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ProgramKey(pub u64);

/// Identity of a configuration (pdi only).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ConfigKey(pub u64);

/// Byte sink for streaming canonical forms (hash, compare, or collect).
trait Sink {
    fn put(&mut self, bytes: &[u8]);
}

struct FnvSink(u64);

impl Sink for FnvSink {
    fn put(&mut self, bytes: &[u8]) {
        self.0 = fnv1a(self.0, bytes);
    }
}

/// Compares the streamed bytes against an expected buffer.
struct CompareSink<'a> {
    expected: &'a [u8],
    pos: usize,
    equal: bool,
}

impl Sink for CompareSink<'_> {
    fn put(&mut self, bytes: &[u8]) {
        if !self.equal {
            return;
        }
        let end = self.pos + bytes.len();
        if end > self.expected.len() || self.expected[self.pos..end] != *bytes {
            self.equal = false;
        }
        self.pos = end;
    }
}

impl Sink for Vec<u8> {
    fn put(&mut self, bytes: &[u8]) {
        self.extend_from_slice(bytes);
    }
}

fn put_kind(sink: &mut impl Sink, kind: BindingKind) {
    let (buf, n) = kind.encode();
    sink.put(&buf[..n]);
}

impl ProgramSpec<'_> {
    /// Lowest masked (Seq/Custom, in-range) word index `>= cursor`, if any.
    fn next_masked_word(&self, cursor: usize) -> Option<usize> {
        let words = self.txn.len() / 4;
        self.dynamic
            .iter()
            .filter(|&&(idx, kind)| kind.masks_txn_word() && idx >= cursor && idx < words)
            .map(|&(idx, _)| idx)
            .min()
    }

    /// Exact byte length of the canonical program layout.
    fn canonical_len(&self) -> usize {
        let dynamic: usize = self
            .dynamic
            .iter()
            .map(|&(_, kind)| 8 + kind.encode().1)
            .sum();
        27 + self.pdi.len() + self.txn.len() + dynamic
    }

    /// Stream the canonical program layout into `sink`.
    fn stream(&self, sink: &mut impl Sink) {
        sink.put(b"P");
        sink.put(&(self.pdi.len() as u64).to_le_bytes());
        sink.put(self.pdi);
        sink.put(b"T");
        sink.put(&(self.txn.len() as u64).to_le_bytes());
        // Static spans verbatim; each masked word is replaced by 4 zero bytes.
        let mut cursor = 0usize;
        while let Some(word) = self.next_masked_word(cursor) {
            sink.put(&self.txn[cursor * 4..word * 4]);
            sink.put(&[0u8; 4]);
            cursor = word + 1;
        }
        sink.put(&self.txn[cursor * 4..]);
        sink.put(b"D");
        sink.put(&(self.dynamic.len() as u64).to_le_bytes());
        for &(idx, kind) in self.dynamic {
            sink.put(&(idx as u64).to_le_bytes());
            put_kind(sink, kind);
        }
    }

    /// Check the in-range inventory contract: every `Seq`/`Custom` index must
    /// address a complete u32 word of `txn`. Does not allocate.
    pub fn validate(&self) -> Result<(), String> {
        let words = self.txn.len() / 4;
        for (n, &(idx, kind)) in self.dynamic.iter().enumerate() {
            if kind.masks_txn_word() && idx >= words {
                return Err(format!(
                    "dynamic[{n}]: {kind:?} word index {idx} out of range (txn has {words} u32 words)"
                ));
            }
        }
        Ok(())
    }

    /// Program identity (see module docs for layout).
    pub fn program_key(&self) -> ProgramKey {
        ProgramKey::from_spec(self)
    }

    /// Config identity (pdi only).
    pub fn config_key(&self) -> ConfigKey {
        ConfigKey::from_pdi(self.pdi)
    }
}

impl ProgramKey {
    /// Hash the canonical program layout. Never panics; out-of-range txn
    /// indices mask nothing (but remain part of the dynamic list).
    pub fn from_spec(spec: &ProgramSpec<'_>) -> Self {
        let mut sink = FnvSink(FNV_OFFSET);
        spec.stream(&mut sink);
        ProgramKey(sink.0)
    }
}

impl ConfigKey {
    /// `fnv1a(FNV_OFFSET, 'C' || pdi)`.
    pub fn from_pdi(pdi: &[u8]) -> Self {
        ConfigKey(fnv1a(fnv1a(FNV_OFFSET, b"C"), pdi))
    }
}

/// Result of a cache lookup.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CacheOutcome {
    /// Existing program found and its canonical bytes matched.
    Hit,
    /// Program built and inserted.
    Miss,
}

pub use CacheOutcome::{Hit, Miss};

struct CacheEntry<V> {
    /// Full canonical program layout (pdi, masked txn, ordered dynamic list).
    canonical: Vec<u8>,
    value: V,
}

/// Cache of built values keyed by [`ProgramKey`], verified by full canonical
/// byte comparison on every hit. Hits do not allocate.
pub struct ProgramCache<V> {
    map: HashMap<ProgramKey, CacheEntry<V>>,
    #[cfg(test)]
    key_fn: fn(&ProgramSpec<'_>) -> ProgramKey,
}

impl<V> Default for ProgramCache<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> ProgramCache<V> {
    /// Empty cache using [`ProgramKey::from_spec`].
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            #[cfg(test)]
            key_fn: ProgramKey::from_spec,
        }
    }

    /// Empty cache with a custom key function (tests: force collisions).
    #[cfg(test)]
    pub(crate) fn with_hasher(key_fn: fn(&ProgramSpec<'_>) -> ProgramKey) -> Self {
        Self {
            map: HashMap::new(),
            key_fn,
        }
    }

    /// Number of cached programs.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True when no programs are cached.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Look up `spec`; on miss call `build` and store the result.
    ///
    /// Errors (builder not called, nothing stored/replaced): invalid dynamic
    /// inventory ([`ProgramSpec::validate`]) or a key collision (same key,
    /// different canonical bytes), reported as `"hash collision: ..."`.
    /// A builder error is propagated and nothing is inserted.
    pub fn get_or_insert_with(
        &mut self,
        spec: ProgramSpec<'_>,
        build: impl FnOnce() -> Result<V, String>,
    ) -> Result<(&mut V, CacheOutcome), String> {
        spec.validate()?;
        #[cfg(test)]
        let key = (self.key_fn)(&spec);
        #[cfg(not(test))]
        let key = ProgramKey::from_spec(&spec);
        match self.map.entry(key) {
            Entry::Occupied(e) => {
                let stored = e.get();
                let mut cmp = CompareSink {
                    expected: &stored.canonical,
                    pos: 0,
                    equal: true,
                };
                spec.stream(&mut cmp);
                if !cmp.equal || cmp.pos != stored.canonical.len() {
                    return Err(format!(
                        "hash collision: program key {:#018x} matches a different canonical program",
                        key.0
                    ));
                }
                Ok((&mut e.into_mut().value, CacheOutcome::Hit))
            }
            Entry::Vacant(v) => {
                let value = build()?;
                let mut canonical = Vec::with_capacity(spec.canonical_len());
                spec.stream(&mut canonical);
                let entry = v.insert(CacheEntry { canonical, value });
                Ok((&mut entry.value, CacheOutcome::Miss))
            }
        }
    }
}

/// One recorded dispatch: program, config, and ordered binding kinds with
/// their current values (values are not part of the tape hash).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TapeEntry {
    pub program: ProgramKey,
    pub config: ConfigKey,
    pub bindings: Vec<(BindingKind, u64)>,
}

/// How an entry is dispatched relative to the previous one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Transition {
    /// First entry, or the config differs from the previous entry.
    Full,
    /// Same [`ConfigKey`] as the immediately previous entry.
    Lean,
}

/// Ordered dispatch record plus its sequence hash (see module docs for layout).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Tape {
    pub entries: Vec<TapeEntry>,
    pub sequence_hash: u64,
}

fn hash_entries(entries: &[TapeEntry]) -> u64 {
    let mut sink = FnvSink(FNV_OFFSET);
    for e in entries {
        sink.put(&e.program.0.to_le_bytes());
        sink.put(&e.config.0.to_le_bytes());
        sink.put(&(e.bindings.len() as u64).to_le_bytes());
        for &(kind, _value) in &e.bindings {
            put_kind(&mut sink, kind);
        }
    }
    sink.0
}

impl Tape {
    /// Record `entries` and compute their sequence hash.
    pub fn record(entries: Vec<TapeEntry>) -> Self {
        let sequence_hash = hash_entries(&entries);
        Tape {
            entries,
            sequence_hash,
        }
    }

    /// Recompute the sequence hash from the current (possibly mutated) entries.
    pub fn actual_hash(&self) -> u64 {
        hash_entries(&self.entries)
    }

    /// Per-entry transitions: `Lean` only when the previous entry has the same config.
    pub fn transitions(&self) -> impl Iterator<Item = Transition> + '_ {
        self.entries.iter().enumerate().map(|(i, e)| {
            if i > 0 && self.entries[i - 1].config == e.config {
                Transition::Lean
            } else {
                Transition::Full
            }
        })
    }

    /// Verify the tape against `recorded_hash`. The hash is recomputed from the
    /// public entries, so entries mutated after [`Tape::record`] (stale
    /// `sequence_hash`) are detected as well.
    pub fn check_replay(&self, recorded_hash: u64) -> Result<(), String> {
        let actual = self.actual_hash();
        if actual != recorded_hash {
            return Err(format!(
                "tape replay mismatch: recorded {recorded_hash:#018x}, actual {actual:#018x}"
            ));
        }
        if actual != self.sequence_hash {
            return Err(format!(
                "tape entries changed since recording: stored {:#018x}, actual {actual:#018x}",
                self.sequence_hash
            ));
        }
        Ok(())
    }
}

/// Verify that `after` differs from `before` only at `declared` word indices.
/// Returns the number of words that actually changed. Rejects length
/// mismatch, out-of-range declared indices, and any undeclared difference.
pub fn verify_patch(before: &[u32], after: &[u32], declared: &[usize]) -> Result<usize, String> {
    if before.len() != after.len() {
        return Err(format!(
            "patch length mismatch: before {} words, after {} words",
            before.len(),
            after.len()
        ));
    }
    if let Some(&d) = declared.iter().find(|&&d| d >= before.len()) {
        return Err(format!(
            "declared patch index {d} out of range ({} words)",
            before.len()
        ));
    }
    let mut changed = 0;
    for (i, (b, a)) in before.iter().zip(after).enumerate() {
        if b != a {
            if !declared.contains(&i) {
                return Err(format!(
                    "undeclared patch at word {i}: {b:#010x} -> {a:#010x}"
                ));
            }
            changed += 1;
        }
    }
    Ok(changed)
}

/// Verify a retained buffer equals a freshly built one; reports the first difference.
pub fn verify_fresh(retained: &[u32], fresh: &[u32]) -> Result<(), String> {
    if retained.len() != fresh.len() {
        return Err(format!(
            "fresh length mismatch: retained {} words, fresh {} words",
            retained.len(),
            fresh.len()
        ));
    }
    match retained.iter().zip(fresh).position(|(r, f)| r != f) {
        Some(i) => Err(format!(
            "fresh mismatch at word {i}: retained {:#010x}, fresh {:#010x}",
            retained[i], fresh[i]
        )),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    const SEQ: BindingKind = BindingKind::Seq { run: 0 };

    #[test]
    fn moving_word_static_vs_dynamic_changes_program_key() {
        let txn = bytes(&[1, 2, 3]);
        let stat = ProgramSpec {
            pdi: b"pdi",
            txn: &txn,
            dynamic: &[],
        };
        let dyn1 = [(1usize, SEQ)];
        let dynamic = ProgramSpec {
            pdi: b"pdi",
            txn: &txn,
            dynamic: &dyn1,
        };
        assert_ne!(stat.program_key(), dynamic.program_key());
        let dyn2 = [(2usize, SEQ)];
        let moved = ProgramSpec {
            pdi: b"pdi",
            txn: &txn,
            dynamic: &dyn2,
        };
        assert_ne!(dynamic.program_key(), moved.program_key());
    }

    #[test]
    fn dynamic_seq_custom_value_changes_keep_key() {
        let dyn_ = [(0usize, SEQ), (2usize, BindingKind::Custom(9))];
        let a = bytes(&[1, 2, 3]);
        let b = bytes(&[0xdead_beef, 2, 0x1234_5678]);
        let ka = ProgramSpec {
            pdi: b"p",
            txn: &a,
            dynamic: &dyn_,
        }
        .program_key();
        let kb = ProgramSpec {
            pdi: b"p",
            txn: &b,
            dynamic: &dyn_,
        }
        .program_key();
        assert_eq!(ka, kb);
        // Static word change does change it.
        let c = bytes(&[1, 7, 3]);
        let kc = ProgramSpec {
            pdi: b"p",
            txn: &c,
            dynamic: &dyn_,
        }
        .program_key();
        assert_ne!(ka, kc);
    }

    #[test]
    fn cmd_bo_indices_do_not_mask_txn() {
        let dyn_ = [
            (0usize, BindingKind::ArgAddress { arg: 1 }),
            (1, BindingKind::InstsAddress),
        ];
        let a = bytes(&[1, 2]);
        let b = bytes(&[9, 2]);
        let ka = ProgramSpec {
            pdi: b"p",
            txn: &a,
            dynamic: &dyn_,
        }
        .program_key();
        let kb = ProgramSpec {
            pdi: b"p",
            txn: &b,
            dynamic: &dyn_,
        }
        .program_key();
        assert_ne!(ka, kb);
    }

    #[test]
    fn unsorted_duplicate_masks_mixed_domains_only_mask_listed_txn_words() {
        // Unsorted, duplicated, adjacent txn masks (1, 2, 4) mixed with cmd-BO
        // indices (0, 3) that must not mask txn words.
        let dyn_ = [
            (4usize, SEQ),
            (1, BindingKind::Custom(2)),
            (0, BindingKind::ArgAddress { arg: 0 }),
            (4, BindingKind::Custom(5)),
            (2, SEQ),
            (3, BindingKind::InstsAddress),
        ];
        let mut base = bytes(&[10, 11, 12, 13, 14, 15]);
        base.extend_from_slice(&[0xaa, 0xbb]); // trailing partial word
        let key = |txn: &[u8]| {
            ProgramSpec {
                pdi: b"p",
                txn,
                dynamic: &dyn_,
            }
            .program_key()
        };
        let base_key = key(&base);

        for word in [1usize, 2, 4] {
            let mut t = base.clone();
            t[word * 4..word * 4 + 4].copy_from_slice(&0x5a5a_5a5au32.to_le_bytes());
            assert_eq!(key(&t), base_key, "masked word {word} must not affect key");
        }
        for pos in [0usize, 1, 3, 8, 12, 13, 15, 20, 23, 24, 25] {
            // pos in static words 0, 3, 5, or trailing bytes 24..26
            let mut t = base.clone();
            t[pos] ^= 0x80;
            if (4..12).contains(&pos) || (16..20).contains(&pos) {
                assert_eq!(key(&t), base_key);
            } else {
                assert_ne!(key(&t), base_key, "static byte {pos} must affect key");
            }
        }
    }

    #[test]
    fn forced_collision_errors_without_calling_builder() {
        let mut cache: ProgramCache<u32> = ProgramCache::with_hasher(|_| ProgramKey(7));
        let t1 = bytes(&[1]);
        let t2 = bytes(&[2]);
        let s1 = ProgramSpec {
            pdi: b"p",
            txn: &t1,
            dynamic: &[],
        };
        let s2 = ProgramSpec {
            pdi: b"p",
            txn: &t2,
            dynamic: &[],
        };
        assert_eq!(cache.get_or_insert_with(s1, || Ok(11)).unwrap().1, Miss);
        let mut called = false;
        let err = cache
            .get_or_insert_with(s2, || {
                called = true;
                Ok(22)
            })
            .map(|_| ())
            .unwrap_err();
        assert!(err.starts_with("hash collision"), "{err}");
        assert!(!called);
        // Original entry not replaced.
        let (v, outcome) = cache.get_or_insert_with(s1, || Err("no".into())).unwrap();
        assert_eq!((*v, outcome), (11, Hit));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_hit_on_dynamic_change_retains_value_static_change_misses() {
        let dyn_ = [(1usize, SEQ)];
        let mut cache: ProgramCache<Vec<u32>> = ProgramCache::default();
        let t1 = bytes(&[1, 2, 3]);
        let (v, o) = cache
            .get_or_insert_with(
                ProgramSpec {
                    pdi: b"p",
                    txn: &t1,
                    dynamic: &dyn_,
                },
                || Ok(vec![1]),
            )
            .unwrap();
        assert_eq!(o, Miss);
        v.push(2);

        let t2 = bytes(&[1, 99, 3]); // only the dynamic word changed
        let mut built = false;
        let (v, o) = cache
            .get_or_insert_with(
                ProgramSpec {
                    pdi: b"p",
                    txn: &t2,
                    dynamic: &dyn_,
                },
                || {
                    built = true;
                    Ok(vec![])
                },
            )
            .unwrap();
        assert_eq!(o, Hit);
        assert!(!built);
        assert_eq!(*v, vec![1, 2]);

        let t3 = bytes(&[5, 99, 3]); // static word changed
        let (v, o) = cache
            .get_or_insert_with(
                ProgramSpec {
                    pdi: b"p",
                    txn: &t3,
                    dynamic: &dyn_,
                },
                || Ok(vec![7]),
            )
            .unwrap();
        assert_eq!(o, Miss);
        assert_eq!(*v, vec![7]);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn builder_error_inserts_nothing() {
        let mut cache: ProgramCache<u8> = ProgramCache::new();
        let t = bytes(&[1]);
        let spec = ProgramSpec {
            pdi: b"p",
            txn: &t,
            dynamic: &[],
        };
        assert_eq!(
            cache
                .get_or_insert_with(spec, || Err("boom".into()))
                .map(|_| ())
                .unwrap_err(),
            "boom"
        );
        assert!(cache.is_empty());
    }

    #[test]
    fn inventory_out_of_range_is_explicit_error() {
        let t = bytes(&[1, 2]);
        let bad = [(2usize, SEQ)];
        let spec = ProgramSpec {
            pdi: b"p",
            txn: &t,
            dynamic: &bad,
        };
        assert!(spec.validate().is_err());
        // Constructor does not panic and masks nothing.
        let _ = spec.program_key();
        let mut cache: ProgramCache<u8> = ProgramCache::new();
        let mut called = false;
        let r = cache.get_or_insert_with(spec, || {
            called = true;
            Ok(0)
        });
        assert!(r.is_err());
        assert!(!called);
        // A partial trailing word is not a valid word either.
        let t5 = vec![0u8; 5];
        let d = [(1usize, BindingKind::Custom(1))];
        assert!(ProgramSpec {
            pdi: b"",
            txn: &t5,
            dynamic: &d
        }
        .validate()
        .is_err());
        // Cmd-BO indices are not range-checked against txn.
        let cmd = [(1000usize, BindingKind::InstsAddress)];
        assert!(ProgramSpec {
            pdi: b"",
            txn: &t5,
            dynamic: &cmd
        }
        .validate()
        .is_ok());
    }

    fn entry(program: u64, config: u64, kinds: &[(BindingKind, u64)]) -> TapeEntry {
        TapeEntry {
            program: ProgramKey(program),
            config: ConfigKey(config),
            bindings: kinds.to_vec(),
        }
    }

    #[test]
    fn stale_program_identity_changes_hash_and_check_replay_errors() {
        let kinds = [(SEQ, 0u64)];
        let txn = bytes(&[1]);
        let changed_txn = bytes(&[2]);
        let original = ProgramSpec {
            pdi: b"p",
            txn: &txn,
            dynamic: &[],
        };
        let changed = ProgramSpec {
            txn: &changed_txn,
            ..original
        };
        let good = Tape::record(vec![entry(
            original.program_key().0,
            original.config_key().0,
            &kinds,
        )]);
        let stale = Tape::record(vec![entry(
            changed.program_key().0,
            changed.config_key().0,
            &kinds,
        )]);
        assert_ne!(good.sequence_hash, stale.sequence_hash);
        assert!(good.check_replay(good.sequence_hash).is_ok());
        assert!(good.check_replay(stale.sequence_hash).is_err());

        // Mutating public entries after record is detected even with the stale stored hash.
        let mut mutated = good.clone();
        mutated.entries[0].program = changed.program_key();
        assert!(mutated.check_replay(good.sequence_hash).is_err());
        assert!(mutated.check_replay(mutated.sequence_hash).is_err());
    }

    #[test]
    fn binding_values_irrelevant_kind_and_order_matter() {
        let base = Tape::record(vec![entry(1, 1, &[(SEQ, 1), (BindingKind::Custom(3), 2)])]);
        let values = Tape::record(vec![entry(
            1,
            1,
            &[(SEQ, 99), (BindingKind::Custom(3), 77)],
        )]);
        assert_eq!(base.sequence_hash, values.sequence_hash);
        let kind = Tape::record(vec![entry(1, 1, &[(SEQ, 1), (BindingKind::Custom(4), 2)])]);
        assert_ne!(base.sequence_hash, kind.sequence_hash);
        let order = Tape::record(vec![entry(1, 1, &[(BindingKind::Custom(3), 2), (SEQ, 1)])]);
        assert_ne!(base.sequence_hash, order.sequence_hash);
        let run = Tape::record(vec![entry(
            1,
            1,
            &[
                (BindingKind::Seq { run: 1 }, 1),
                (BindingKind::Custom(3), 2),
            ],
        )]);
        assert_ne!(base.sequence_hash, run.sequence_hash);
    }

    #[test]
    fn transitions_full_lean_by_previous_config() {
        let t = Tape::record(
            [1u64, 1, 2, 2, 1]
                .iter()
                .map(|&c| entry(c, c, &[]))
                .collect(),
        );
        let got: Vec<_> = t.transitions().collect();
        use Transition::{Full, Lean};
        assert_eq!(got, vec![Full, Lean, Full, Lean, Full]);
    }

    #[test]
    fn empty_tape() {
        let t = Tape::record(vec![]);
        assert_eq!(t.sequence_hash, FNV_OFFSET);
        assert_eq!(t.transitions().count(), 0);
        assert!(t.check_replay(FNV_OFFSET).is_ok());
        assert!(t.check_replay(0).is_err());
    }

    #[test]
    fn verify_patch_declared_and_undeclared() {
        let before = [1u32, 2, 3, 4];
        let after = [1u32, 9, 3, 8];
        assert_eq!(verify_patch(&before, &after, &[1, 3]), Ok(2));
        // Declared but unchanged words are not counted.
        assert_eq!(verify_patch(&before, &after, &[0, 1, 3]), Ok(2));
        assert!(verify_patch(&before, &after, &[1]).is_err());
        assert!(verify_patch(&before, &after[..3], &[1]).is_err());
        assert!(verify_patch(&before, &before, &[4]).is_err());
        assert_eq!(verify_patch(&[], &[], &[]), Ok(0));
    }

    #[test]
    fn verify_fresh_rejects_one_bit_and_length() {
        let a = [1u32, 2, 3];
        assert!(verify_fresh(&a, &a).is_ok());
        let mut b = a;
        b[2] ^= 1 << 17;
        assert!(verify_fresh(&a, &b).is_err());
        assert!(verify_fresh(&a, &a[..2]).is_err());
        assert!(verify_fresh(&[], &[]).is_ok());
    }

    #[test]
    fn fnv1a_known_vectors_and_layout_contract() {
        assert_eq!(fnv1a(FNV_OFFSET, b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(FNV_OFFSET, b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(ConfigKey::from_pdi(b"xy").0, fnv1a(FNV_OFFSET, b"Cxy"));
        // Program layout: P|len|pdi|T|len|txn|D|count
        let mut raw = Vec::new();
        raw.extend_from_slice(b"P");
        raw.extend_from_slice(&2u64.to_le_bytes());
        raw.extend_from_slice(b"ab");
        raw.extend_from_slice(b"T");
        raw.extend_from_slice(&4u64.to_le_bytes());
        raw.extend_from_slice(&[0, 0, 0, 0]); // masked
        raw.extend_from_slice(b"D");
        raw.extend_from_slice(&1u64.to_le_bytes());
        raw.extend_from_slice(&0u64.to_le_bytes());
        raw.extend_from_slice(&[3, 5, 0, 0, 0]);
        let t = bytes(&[0xffff_ffff]);
        let d = [(0usize, BindingKind::Seq { run: 5 })];
        let key = ProgramSpec {
            pdi: b"ab",
            txn: &t,
            dynamic: &d,
        }
        .program_key();
        assert_eq!(key.0, fnv1a(FNV_OFFSET, &raw));
    }
}
