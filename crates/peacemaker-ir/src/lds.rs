// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::cfg::InstId;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LdsKind {
    Load,
    Store,
    Atomic,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddrFact {
    Const(u32),
    Affine { base: u32, lane_stride_max: u32 },
    Bounded { lo: u32, hi: u32 },
    Unknown,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SegId(pub usize);
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SegmentOrigin {
    Declared { name: String },
    Inferred,
    Whole,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LdsSegment {
    pub id: SegId,
    pub range: Option<(u32, u32)>,
    pub origin: SegmentOrigin,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LdsAccess {
    pub inst: InstId,
    pub addr: AddrFact,
    pub bytes: u32,
    pub kind: LdsKind,
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LdsFacts {
    pub segments: Vec<LdsSegment>,
    pub accesses: Vec<(LdsAccess, SegId)>,
    pub max_end: Option<u32>,
}
