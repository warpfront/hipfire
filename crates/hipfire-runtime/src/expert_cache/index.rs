// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fixed-capacity open-addressing map from [`ExpertKey`] to a `(pool, slot)` location.
//!
//! `std::collections::HashMap` can reallocate under insert/remove churn when it is nearly full
//! (tombstone rehash), so the cache's "no steady-state allocation" rule needs its own table: the
//! bucket array is sized at construction (a power of two at least twice the entry bound), probing
//! is linear, and removal shifts the following cluster back (no tombstones). An insert beyond the
//! entry bound is a refusal, never growth.

use super::ExpertKey;

/// Where a key lives: the pool and the slot inside it (or, for the use log, a position).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Loc {
    pub pool: u16,
    pub slot: u32,
}

#[derive(Clone, Copy)]
struct Bucket {
    key: ExpertKey,
    loc: Loc,
    used: bool,
}

const EMPTY: Bucket = Bucket {
    key: ExpertKey {
        owner_id: 0,
        owner_generation: 0,
        group_id: 0,
        expert_id: 0,
    },
    loc: Loc { pool: 0, slot: 0 },
    used: false,
};

/// The map. `len() <= bound` always holds.
pub(crate) struct KeyIndex {
    buckets: Box<[Bucket]>,
    mask: usize,
    len: usize,
    bound: usize,
}

/// `insert` beyond the construction bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IndexFull;

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl ExpertKey {
    fn hash64(&self) -> u64 {
        let a = mix(self.owner_id ^ 0x9e37_79b9_7f4a_7c15);
        let b = mix(self.owner_generation.wrapping_add(a));
        let c = mix(((self.group_id as u64) << 32 | self.expert_id as u64) ^ b);
        mix(c ^ a.rotate_left(17))
    }
}

impl KeyIndex {
    /// A table that holds up to `bound` entries (`bound` may be 0).
    pub(crate) fn new(bound: usize) -> Self {
        let buckets = (bound.saturating_mul(2)).max(4).next_power_of_two();
        Self {
            buckets: vec![EMPTY; buckets].into_boxed_slice(),
            mask: buckets - 1,
            len: 0,
            bound,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn bound(&self) -> usize {
        self.bound
    }

    /// Bucket-array length (a capacity probe for the no-growth tests).
    #[cfg(test)]
    pub(crate) fn buckets(&self) -> usize {
        self.buckets.len()
    }

    fn home(&self, key: &ExpertKey) -> usize {
        key.hash64() as usize & self.mask
    }

    fn find(&self, key: &ExpertKey) -> Option<usize> {
        let mut at = self.home(key);
        // `len <= bound <= buckets / 2`, so an empty bucket always ends the probe.
        loop {
            let b = &self.buckets[at];
            if !b.used {
                return None;
            }
            if b.key == *key {
                return Some(at);
            }
            at = (at + 1) & self.mask;
        }
    }

    pub(crate) fn get(&self, key: &ExpertKey) -> Option<Loc> {
        self.find(key).map(|at| self.buckets[at].loc)
    }

    /// Insert a new key. An existing key is overwritten in place (and does not count again).
    pub(crate) fn insert(&mut self, key: ExpertKey, loc: Loc) -> Result<(), IndexFull> {
        if let Some(at) = self.find(&key) {
            self.buckets[at].loc = loc;
            return Ok(());
        }
        if self.len >= self.bound {
            return Err(IndexFull);
        }
        let mut at = self.home(&key);
        while self.buckets[at].used {
            at = (at + 1) & self.mask;
        }
        self.buckets[at] = Bucket {
            key,
            loc,
            used: true,
        };
        self.len += 1;
        Ok(())
    }

    /// Remove `key` (backward-shift deletion: no tombstones, no allocation).
    pub(crate) fn remove(&mut self, key: &ExpertKey) -> Option<Loc> {
        let mut hole = self.find(key)?;
        let loc = self.buckets[hole].loc;
        let mut next = (hole + 1) & self.mask;
        while self.buckets[next].used {
            let home = self.home(&self.buckets[next].key);
            // `next` may move into `hole` unless its home lies cyclically in (hole, next].
            let between = if hole <= next {
                hole < home && home <= next
            } else {
                hole < home || home <= next
            };
            if !between {
                self.buckets[hole] = self.buckets[next];
                hole = next;
            }
            next = (next + 1) & self.mask;
        }
        self.buckets[hole].used = false;
        self.len -= 1;
        Some(loc)
    }

    pub(crate) fn clear(&mut self) {
        for b in self.buckets.iter_mut() {
            b.used = false;
        }
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn key(n: u64) -> ExpertKey {
        ExpertKey {
            owner_id: n % 3,
            owner_generation: n / 7,
            group_id: (n % 5) as u32,
            expert_id: n as u32,
        }
    }

    #[test]
    fn matches_a_hashmap_under_churn_without_growing() {
        let mut index = KeyIndex::new(48);
        let buckets = index.buckets();
        let mut model: HashMap<ExpertKey, Loc> = HashMap::new();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for step in 0..20_000u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let k = key(state % 200);
            if state % 3 == 0 {
                assert_eq!(index.remove(&k), model.remove(&k), "step {step}");
            } else if model.len() < 48 || model.contains_key(&k) {
                let loc = Loc {
                    pool: (state % 4) as u16,
                    slot: step as u32,
                };
                index.insert(k, loc).unwrap();
                model.insert(k, loc);
            } else {
                assert_eq!(index.insert(k, Loc { pool: 0, slot: 0 }), Err(IndexFull));
            }
            assert_eq!(index.len(), model.len());
            assert_eq!(index.get(&k), model.get(&k).copied());
        }
        assert_eq!(index.buckets(), buckets, "the table never grows");
        for (k, loc) in &model {
            assert_eq!(index.get(k), Some(*loc));
        }
    }

    #[test]
    fn zero_bound_refuses_every_insert() {
        let mut index = KeyIndex::new(0);
        assert_eq!(
            index.insert(key(1), Loc { pool: 0, slot: 0 }),
            Err(IndexFull)
        );
        assert_eq!(index.get(&key(1)), None);
    }
}
