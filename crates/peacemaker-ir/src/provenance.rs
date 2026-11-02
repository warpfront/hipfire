// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct EditId(pub u64);
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ObjectSha(pub [u8; 32]);
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TextSha(pub [u8; 32]);
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Source {
    Object(ObjectSha),
    Text(TextSha),
    Builder { kernel_id: String, variant: String },
    Edit(EditId),
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Provenance {
    pub source: Source,
    pub pc: Option<u32>,
    pub bytes: Option<[u32; 3]>,
    pub line: Option<u32>,
    pub edit: Option<EditId>,
}
impl Default for Provenance {
    fn default() -> Self {
        Self {
            source: Source::Builder {
                kernel_id: String::new(),
                variant: String::new(),
            },
            pc: None,
            bytes: None,
            line: None,
            edit: None,
        }
    }
}
