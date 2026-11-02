// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::inst::Inst;
use smallvec::SmallVec;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct InstId(pub usize);
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct BlockId(pub usize);
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BarrierId(pub u32);
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Cond {
    Scc0,
    Scc1,
    Vccz,
    Vccnz,
    Execz,
    Execnz,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BarrierKind {
    Full,
    Signal(BarrierId),
    Wait,
    Named(u16),
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Terminator {
    FallThrough,
    Jump(BlockId),
    Branch {
        cond: Cond,
        taken: BlockId,
        fallthrough: BlockId,
    },
    EndPgm,
    Unreachable,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Block {
    pub id: BlockId,
    pub range: (usize, usize),
    pub term: Terminator,
    pub preds: SmallVec<[BlockId; 2]>,
    pub succs: SmallVec<[BlockId; 2]>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Arena<T> {
    slots: Vec<Option<T>>,
}
impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T> Arena<T> {
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }
    pub fn insert(&mut self, value: T) -> InstId {
        let id = InstId(self.slots.len());
        self.slots.push(Some(value));
        id
    }
    pub fn get(&self, id: InstId) -> Option<&T> {
        self.slots.get(id.0)?.as_ref()
    }
    pub fn get_mut(&mut self, id: InstId) -> Option<&mut T> {
        self.slots.get_mut(id.0)?.as_mut()
    }
    pub fn remove(&mut self, id: InstId) -> Option<T> {
        self.slots.get_mut(id.0)?.take()
    }
    pub fn len(&self) -> usize {
        self.slots.len()
    }
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = (InstId, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| Some((InstId(id), slot.as_ref()?)))
    }
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Body {
    pub insts: Arena<Inst>,
    pub blocks: Vec<Block>,
    pub layout: Vec<InstId>,
}
