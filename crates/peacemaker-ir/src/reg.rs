// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::cfg::{BlockId, InstId};
use std::fmt;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Kind {
    V,
    S,
    Ttmp,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct RegRef {
    pub kind: Kind,
    pub base: u16,
    pub len: u8,
}
impl RegRef {
    pub fn validate(self) -> Result<(), String> {
        if self.len == 0 {
            return Err("empty register range".into());
        }
        let end = self
            .base
            .checked_add(u16::from(self.len))
            .ok_or("register range overflow")?;
        let limit = match self.kind {
            Kind::V => 256,
            Kind::S => 106,
            Kind::Ttmp => 16,
        };
        if end > limit {
            return Err(format!("register range exceeds {limit}"));
        }
        Ok(())
    }
    pub fn overlaps(self, other: Self) -> bool {
        self.kind == other.kind
            && self.base < other.base + u16::from(other.len)
            && other.base < self.base + u16::from(self.len)
    }
}
impl fmt::Display for RegRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix = match self.kind {
            Kind::V => "v",
            Kind::S => "s",
            Kind::Ttmp => "ttmp",
        };
        if self.len == 1 {
            write!(f, "{prefix}{}", self.base)
        } else {
            write!(
                f,
                "{prefix}[{}:{}]",
                self.base,
                self.base + u16::from(self.len) - 1
            )
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct RegSet(pub Vec<RegRef>);
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Scope {
    Whole,
    Blocks(Vec<BlockId>),
    Between(InstId, InstId),
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClaimOwner {
    Builder,
    Edit(crate::provenance::EditId),
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegClaim {
    pub name: String,
    pub reg: RegRef,
    pub scope: Scope,
    pub owner: ClaimOwner,
}
