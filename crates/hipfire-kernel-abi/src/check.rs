// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Contract vs evidence: compare a [`KernelAbiV1`] export with plain data
//! gathered from a generator spec, an object's AMDGPU metadata or a railgun
//! receipt, and report every disagreement as a [`Drift`].

use core::fmt;

use serde::Serialize;

use crate::validate::block_product;
use crate::{Arch, ExportV1, FieldKind, KernelAbiV1, ObjectRefV1};

/// Where a piece of evidence came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSource {
    Generator,
    ObjectMetadata,
    RailgunReceipt,
}

/// One kernel argument as the evidence describes it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ArgEvidence {
    pub name: String,
    pub offset: u32,
    pub size: u32,
    /// AMDGPU `.value_kind` spelling (`global_buffer`, `by_value`,
    /// `hidden_block_count_x`, ...).
    pub value_kind: String,
    /// `.actual_access` spelling (`read_only`, ...) when the evidence has one.
    pub access: Option<String>,
}

/// Plain-data description of one export, as one evidence source sees it.
/// Every `Option` is compared only when present.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExportEvidence {
    pub source: EvidenceSource,
    pub arch: Option<Arch>,
    pub symbol: String,
    /// Explicit and hidden arguments, in any order.
    pub args: Vec<ArgEvidence>,
    pub segment_bytes: Option<u32>,
    pub segment_align: Option<u32>,
    pub max_flat_workgroup_size: Option<u32>,
    pub threads: Option<u32>,
    pub dynamic_lds: Option<u32>,
    /// `group_segment_fixed_size`.
    pub group_segment_fixed: Option<u32>,
    pub wave: Option<u32>,
    pub object_sha256: Option<String>,
}

/// One disagreement between a contract export and evidence. Every variant
/// names the contract's export `symbol`; `contract` / `evidence` carry the
/// two disagreeing values.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Drift {
    /// The evidence is for another symbol.
    Symbol { symbol: String, evidence: String },
    Arch {
        symbol: String,
        contract: Arch,
        evidence: Arch,
    },
    /// The contract has an argument the evidence lacks.
    MissingArg {
        symbol: String,
        name: String,
        offset: u32,
        value_kind: String,
    },
    /// The evidence has an argument the contract lacks.
    ExtraArg {
        symbol: String,
        name: String,
        offset: u32,
        value_kind: String,
    },
    ArgName {
        symbol: String,
        offset: u32,
        contract: String,
        evidence: String,
    },
    ArgWidth {
        symbol: String,
        offset: u32,
        name: String,
        contract: u32,
        evidence: u32,
    },
    ArgKind {
        symbol: String,
        offset: u32,
        name: String,
        contract: String,
        evidence: String,
    },
    ArgAccess {
        symbol: String,
        offset: u32,
        name: String,
        contract: String,
        evidence: String,
    },
    /// Kernarg segment size.
    Segment {
        symbol: String,
        contract: u32,
        evidence: u32,
    },
    SegmentAlign {
        symbol: String,
        contract: u32,
        evidence: u32,
    },
    /// `max_flat_workgroup_size` below the contract's block product.
    Workgroup {
        symbol: String,
        /// Contract block product.
        contract: u32,
        /// Evidence max flat workgroup size.
        evidence: u32,
    },
    /// Evidence thread count differs from the contract's block product.
    Threads {
        symbol: String,
        /// Contract block product.
        contract: u32,
        evidence: u32,
    },
    DynamicLds {
        symbol: String,
        contract: u32,
        evidence: u32,
    },
    /// Static LDS (`group_segment_fixed_size`).
    StaticLds {
        symbol: String,
        contract: u32,
        evidence: u32,
    },
    Wave {
        symbol: String,
        contract: u32,
        evidence: u32,
    },
    ObjectSha {
        symbol: String,
        contract: String,
        evidence: String,
    },
}

impl fmt::Display for Drift {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Drift::Symbol { symbol, evidence } => {
                write!(f, "{symbol}: evidence is for symbol {evidence}")
            }
            Drift::Arch {
                symbol,
                contract,
                evidence,
            } => write!(
                f,
                "{symbol}: arch contract {} vs evidence {}",
                contract.name(),
                evidence.name()
            ),
            Drift::MissingArg {
                symbol,
                name,
                offset,
                value_kind,
            } => write!(
                f,
                "{symbol}: contract arg {name} @{offset} ({value_kind}) is missing from the evidence"
            ),
            Drift::ExtraArg {
                symbol,
                name,
                offset,
                value_kind,
            } => write!(
                f,
                "{symbol}: evidence arg {name} @{offset} ({value_kind}) is not in the contract"
            ),
            Drift::ArgName {
                symbol,
                offset,
                contract,
                evidence,
            } => write!(
                f,
                "{symbol}: arg @{offset} name contract {contract} vs evidence {evidence}"
            ),
            Drift::ArgWidth {
                symbol,
                offset,
                name,
                contract,
                evidence,
            } => write!(
                f,
                "{symbol}: arg {name} @{offset} width contract {contract} vs evidence {evidence}"
            ),
            Drift::ArgKind {
                symbol,
                offset,
                name,
                contract,
                evidence,
            } => write!(
                f,
                "{symbol}: arg {name} @{offset} value_kind contract {contract} vs evidence {evidence}"
            ),
            Drift::ArgAccess {
                symbol,
                offset,
                name,
                contract,
                evidence,
            } => write!(
                f,
                "{symbol}: arg {name} @{offset} access contract {contract} vs evidence {evidence}"
            ),
            Drift::Segment {
                symbol,
                contract,
                evidence,
            } => write!(
                f,
                "{symbol}: kernarg segment size contract {contract} vs evidence {evidence}"
            ),
            Drift::SegmentAlign {
                symbol,
                contract,
                evidence,
            } => write!(
                f,
                "{symbol}: kernarg segment align contract {contract} vs evidence {evidence}"
            ),
            Drift::Workgroup {
                symbol,
                contract,
                evidence,
            } => write!(
                f,
                "{symbol}: max flat workgroup size {evidence} is below the contract block product {contract}"
            ),
            Drift::Threads {
                symbol,
                contract,
                evidence,
            } => write!(
                f,
                "{symbol}: threads contract {contract} vs evidence {evidence}"
            ),
            Drift::DynamicLds {
                symbol,
                contract,
                evidence,
            } => write!(
                f,
                "{symbol}: dynamic LDS contract {contract} vs evidence {evidence}"
            ),
            Drift::StaticLds {
                symbol,
                contract,
                evidence,
            } => write!(
                f,
                "{symbol}: static LDS contract {contract} vs evidence {evidence}"
            ),
            Drift::Wave {
                symbol,
                contract,
                evidence,
            } => write!(
                f,
                "{symbol}: wave contract {contract} vs evidence {evidence}"
            ),
            Drift::ObjectSha {
                symbol,
                contract,
                evidence,
            } => write!(
                f,
                "{symbol}: object sha256 contract {contract} vs evidence {evidence}"
            ),
        }
    }
}

/// One contract argument (explicit or hidden) in comparable form.
struct ContractArg {
    name: String,
    offset: u32,
    width: u32,
    value_kind: String,
    /// Pointer operands only.
    access: Option<&'static str>,
    hidden: bool,
}

fn contract_args(abi: &KernelAbiV1) -> Vec<ContractArg> {
    let mut args = Vec::with_capacity(abi.fields.len() + abi.hidden.len());
    for f in abi.fields.iter() {
        let (value_kind, access) = match f.kind {
            FieldKind::Pointer { operand } => (
                "global_buffer",
                abi.operand_index(operand)
                    .map(|i| abi.operands[i].access.metadata_name()),
            ),
            FieldKind::Scalar { .. } => ("by_value", None),
        };
        args.push(ContractArg {
            name: f.name.to_string(),
            offset: f.offset,
            width: f.width,
            value_kind: value_kind.to_string(),
            access,
            hidden: false,
        });
    }
    for h in abi.hidden.iter() {
        let value_kind = h.kind.value_kind();
        args.push(ContractArg {
            name: value_kind.clone(),
            offset: h.offset,
            width: h.width,
            value_kind,
            access: None,
            hidden: true,
        });
    }
    args
}

fn to_u32(v: u64) -> u32 {
    u32::try_from(v).unwrap_or(u32::MAX)
}

/// Compare `export` (of `abi`) with `evidence`; every disagreement is
/// returned, an empty vector means the evidence agrees with the contract.
///
/// Arguments match by offset; hidden words match by offset and value kind.
/// An access is compared only when the evidence carries one. The object
/// digest is compared only for [`ObjectRefV1::Embedded`] exports.
pub fn check_export(abi: &KernelAbiV1, export: &ExportV1, evidence: &ExportEvidence) -> Vec<Drift> {
    let symbol = || export.symbol.to_string();
    let mut drifts = Vec::new();

    if evidence.symbol != export.symbol {
        drifts.push(Drift::Symbol {
            symbol: symbol(),
            evidence: evidence.symbol.clone(),
        });
    }
    if let Some(arch) = evidence.arch {
        if arch != export.arch {
            drifts.push(Drift::Arch {
                symbol: symbol(),
                contract: export.arch,
                evidence: arch,
            });
        }
    }

    let mut used = vec![false; evidence.args.len()];
    for c in contract_args(abi) {
        let found = evidence.args.iter().enumerate().find(|(i, a)| {
            !used[*i] && a.offset == c.offset && (!c.hidden || a.value_kind == c.value_kind)
        });
        let Some((i, a)) = found else {
            drifts.push(Drift::MissingArg {
                symbol: symbol(),
                name: c.name,
                offset: c.offset,
                value_kind: c.value_kind,
            });
            continue;
        };
        used[i] = true;
        if !c.hidden && a.name != c.name {
            drifts.push(Drift::ArgName {
                symbol: symbol(),
                offset: c.offset,
                contract: c.name.clone(),
                evidence: a.name.clone(),
            });
        }
        if a.size != c.width {
            drifts.push(Drift::ArgWidth {
                symbol: symbol(),
                offset: c.offset,
                name: c.name.clone(),
                contract: c.width,
                evidence: a.size,
            });
        }
        if a.value_kind != c.value_kind {
            drifts.push(Drift::ArgKind {
                symbol: symbol(),
                offset: c.offset,
                name: c.name.clone(),
                contract: c.value_kind.clone(),
                evidence: a.value_kind.clone(),
            });
        }
        if let (Some(want), Some(got)) = (c.access, a.access.as_deref()) {
            if want != got {
                drifts.push(Drift::ArgAccess {
                    symbol: symbol(),
                    offset: c.offset,
                    name: c.name.clone(),
                    contract: want.to_string(),
                    evidence: got.to_string(),
                });
            }
        }
    }
    for (a, _) in evidence.args.iter().zip(&used).filter(|(_, u)| !**u) {
        drifts.push(Drift::ExtraArg {
            symbol: symbol(),
            name: a.name.clone(),
            offset: a.offset,
            value_kind: a.value_kind.clone(),
        });
    }

    if let Some(v) = evidence.segment_bytes {
        if v != abi.segment_bytes {
            drifts.push(Drift::Segment {
                symbol: symbol(),
                contract: abi.segment_bytes,
                evidence: v,
            });
        }
    }
    if let Some(v) = evidence.segment_align {
        if v != abi.segment_align {
            drifts.push(Drift::SegmentAlign {
                symbol: symbol(),
                contract: abi.segment_align,
                evidence: v,
            });
        }
    }

    let threads = block_product(export.block);
    if let Some(v) = evidence.max_flat_workgroup_size {
        if u64::from(v) < threads {
            drifts.push(Drift::Workgroup {
                symbol: symbol(),
                contract: to_u32(threads),
                evidence: v,
            });
        }
    }
    if let Some(v) = evidence.threads {
        if u64::from(v) != threads {
            drifts.push(Drift::Threads {
                symbol: symbol(),
                contract: to_u32(threads),
                evidence: v,
            });
        }
    }
    if let Some(v) = evidence.dynamic_lds {
        if v != export.dynamic_lds {
            drifts.push(Drift::DynamicLds {
                symbol: symbol(),
                contract: export.dynamic_lds,
                evidence: v,
            });
        }
    }
    if let Some(v) = evidence.group_segment_fixed {
        if v != export.group_segment_fixed {
            drifts.push(Drift::StaticLds {
                symbol: symbol(),
                contract: export.group_segment_fixed,
                evidence: v,
            });
        }
    }
    if let Some(v) = evidence.wave {
        if v != abi.wave {
            drifts.push(Drift::Wave {
                symbol: symbol(),
                contract: abi.wave,
                evidence: v,
            });
        }
    }
    if let (ObjectRefV1::Embedded { sha256, .. }, Some(got)) =
        (export.object, evidence.object_sha256.as_deref())
    {
        if !got.eq_ignore_ascii_case(sha256) {
            drifts.push(Drift::ObjectSha {
                symbol: symbol(),
                contract: sha256.to_string(),
                evidence: got.to_string(),
            });
        }
    }

    drifts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validate::toy::TOY;

    fn evidence(abi: &KernelAbiV1, e: &ExportV1) -> ExportEvidence {
        let mut args = Vec::new();
        for f in abi.fields {
            let (value_kind, access) = match f.kind {
                FieldKind::Pointer { operand } => (
                    "global_buffer",
                    Some(
                        abi.operands[abi.operand_index(operand).unwrap()]
                            .access
                            .metadata_name()
                            .to_string(),
                    ),
                ),
                FieldKind::Scalar { .. } => ("by_value", None),
            };
            args.push(ArgEvidence {
                name: f.name.to_string(),
                offset: f.offset,
                size: f.width,
                value_kind: value_kind.to_string(),
                access,
            });
        }
        for h in abi.hidden {
            args.push(ArgEvidence {
                name: String::new(),
                offset: h.offset,
                size: h.width,
                value_kind: h.kind.value_kind(),
                access: None,
            });
        }
        ExportEvidence {
            source: EvidenceSource::Generator,
            arch: Some(e.arch),
            symbol: e.symbol.to_string(),
            args,
            segment_bytes: Some(abi.segment_bytes),
            segment_align: Some(abi.segment_align),
            max_flat_workgroup_size: Some(1024),
            threads: Some(to_u32(block_product(e.block))),
            dynamic_lds: Some(e.dynamic_lds),
            group_segment_fixed: Some(e.group_segment_fixed),
            wave: Some(abi.wave),
            object_sha256: match e.object {
                ObjectRefV1::Embedded { sha256, .. } => Some(sha256.to_string()),
                ObjectRefV1::HipccSource { .. } => None,
            },
        }
    }

    fn arg_mut(ev: &mut ExportEvidence, offset: u32) -> &mut ArgEvidence {
        ev.args.iter_mut().find(|a| a.offset == offset).unwrap()
    }

    fn check(mutate: impl FnOnce(&mut ExportEvidence)) -> Vec<Drift> {
        let e = &TOY.exports[0];
        let mut ev = evidence(&TOY, e);
        mutate(&mut ev);
        check_export(&TOY, e, &ev)
    }

    #[test]
    fn matching_evidence_has_no_drift_for_every_export() {
        for e in TOY.exports {
            assert_eq!(
                check_export(&TOY, e, &evidence(&TOY, e)),
                vec![],
                "{}",
                e.symbol
            );
        }
    }

    #[test]
    fn evidence_order_does_not_matter() {
        assert_eq!(check(|ev| ev.args.reverse()), vec![]);
    }

    #[test]
    fn symbol_and_arch() {
        assert_eq!(
            check(|ev| ev.symbol = "other".into()),
            vec![Drift::Symbol {
                symbol: "toy_a".into(),
                evidence: "other".into()
            }]
        );
        assert_eq!(
            check(|ev| ev.arch = Some(Arch::Gfx1201)),
            vec![Drift::Arch {
                symbol: "toy_a".into(),
                contract: Arch::Gfx1100,
                evidence: Arch::Gfx1201
            }]
        );
        assert_eq!(check(|ev| ev.arch = None), vec![]);
    }

    #[test]
    fn dropped_arg_is_missing() {
        assert_eq!(
            check(|ev| ev.args.retain(|a| a.offset != 24)),
            vec![Drift::MissingArg {
                symbol: "toy_a".into(),
                name: "k".into(),
                offset: 24,
                value_kind: "by_value".into()
            }]
        );
    }

    #[test]
    fn dropped_hidden_word_is_missing_by_value_kind() {
        assert_eq!(
            check(|ev| ev.args.retain(|a| a.value_kind != "hidden_block_count_x")),
            vec![Drift::MissingArg {
                symbol: "toy_a".into(),
                name: "hidden_block_count_x".into(),
                offset: 40,
                value_kind: "hidden_block_count_x".into()
            }]
        );
        // Same offset, other hidden kind: the contract word is missing and the
        // evidence word is extra.
        let d = check(|ev| arg_mut(ev, 40).value_kind = "hidden_block_count_y".into());
        assert_eq!(d.len(), 2, "{d:?}");
        assert!(matches!(&d[0], Drift::MissingArg { offset: 40, .. }));
        assert!(matches!(&d[1], Drift::ExtraArg { offset: 40, .. }));
    }

    #[test]
    fn extra_arg() {
        assert_eq!(
            check(|ev| ev.args.push(ArgEvidence {
                name: "zz".into(),
                offset: 200,
                size: 8,
                value_kind: "global_buffer".into(),
                access: None
            })),
            vec![Drift::ExtraArg {
                symbol: "toy_a".into(),
                name: "zz".into(),
                offset: 200,
                value_kind: "global_buffer".into()
            }]
        );
        // A duplicate at an already matched offset is extra too.
        let d = check(|ev| {
            let dup = ev.args[2].clone();
            ev.args.push(dup);
        });
        assert!(
            matches!(d.as_slice(), [Drift::ExtraArg { offset: 16, .. }]),
            "{d:?}"
        );
    }

    #[test]
    fn renamed_arg() {
        assert_eq!(
            check(|ev| arg_mut(ev, 16).name = "mm".into()),
            vec![Drift::ArgName {
                symbol: "toy_a".into(),
                offset: 16,
                contract: "m".into(),
                evidence: "mm".into()
            }]
        );
    }

    #[test]
    fn moved_offset_is_missing_plus_extra() {
        let d = check(|ev| arg_mut(ev, 16).offset = 17);
        assert_eq!(
            d,
            vec![
                Drift::MissingArg {
                    symbol: "toy_a".into(),
                    name: "m".into(),
                    offset: 16,
                    value_kind: "by_value".into()
                },
                Drift::ExtraArg {
                    symbol: "toy_a".into(),
                    name: "m".into(),
                    offset: 17,
                    value_kind: "by_value".into()
                },
            ]
        );
    }

    #[test]
    fn wrong_width() {
        assert_eq!(
            check(|ev| arg_mut(ev, 16).size = 8),
            vec![Drift::ArgWidth {
                symbol: "toy_a".into(),
                offset: 16,
                name: "m".into(),
                contract: 4,
                evidence: 8
            }]
        );
        let d = check(|ev| arg_mut(ev, 52).size = 4);
        assert!(
            matches!(d.as_slice(), [Drift::ArgWidth { offset: 52, .. }]),
            "{d:?}"
        );
    }

    #[test]
    fn wrong_value_kind() {
        assert_eq!(
            check(|ev| arg_mut(ev, 16).value_kind = "global_buffer".into()),
            vec![Drift::ArgKind {
                symbol: "toy_a".into(),
                offset: 16,
                name: "m".into(),
                contract: "by_value".into(),
                evidence: "global_buffer".into()
            }]
        );
    }

    #[test]
    fn wrong_access_only_when_evidence_has_one() {
        assert_eq!(
            check(|ev| arg_mut(ev, 0).access = Some("write_only".into())),
            vec![Drift::ArgAccess {
                symbol: "toy_a".into(),
                offset: 0,
                name: "x".into(),
                contract: "read_only".into(),
                evidence: "write_only".into()
            }]
        );
        assert_eq!(check(|ev| arg_mut(ev, 0).access = None), vec![]);
        // Scalars have no contract access: evidence access is not compared.
        assert_eq!(
            check(|ev| arg_mut(ev, 16).access = Some("read_only".into())),
            vec![]
        );
    }

    #[test]
    fn segment_and_alignment() {
        assert_eq!(
            check(|ev| ev.segment_bytes = Some(112)),
            vec![Drift::Segment {
                symbol: "toy_a".into(),
                contract: 104,
                evidence: 112
            }]
        );
        assert_eq!(
            check(|ev| ev.segment_align = Some(16)),
            vec![Drift::SegmentAlign {
                symbol: "toy_a".into(),
                contract: 8,
                evidence: 16
            }]
        );
        assert_eq!(
            check(|ev| {
                ev.segment_bytes = None;
                ev.segment_align = None;
            }),
            vec![]
        );
    }

    #[test]
    fn workgroup_and_threads() {
        // Block is 64x1x1.
        assert_eq!(
            check(|ev| ev.max_flat_workgroup_size = Some(32)),
            vec![Drift::Workgroup {
                symbol: "toy_a".into(),
                contract: 64,
                evidence: 32
            }]
        );
        assert_eq!(check(|ev| ev.max_flat_workgroup_size = Some(64)), vec![]);
        assert_eq!(
            check(|ev| ev.threads = Some(128)),
            vec![Drift::Threads {
                symbol: "toy_a".into(),
                contract: 64,
                evidence: 128
            }]
        );
        assert_eq!(check(|ev| ev.threads = None), vec![]);
    }

    #[test]
    fn dynamic_and_static_lds() {
        assert_eq!(
            check(|ev| ev.dynamic_lds = Some(16)),
            vec![Drift::DynamicLds {
                symbol: "toy_a".into(),
                contract: 0,
                evidence: 16
            }]
        );
        assert_eq!(
            check(|ev| ev.group_segment_fixed = Some(2048)),
            vec![Drift::StaticLds {
                symbol: "toy_a".into(),
                contract: 1024,
                evidence: 2048
            }]
        );
    }

    #[test]
    fn wave() {
        assert_eq!(
            check(|ev| ev.wave = Some(64)),
            vec![Drift::Wave {
                symbol: "toy_a".into(),
                contract: 32,
                evidence: 64
            }]
        );
    }

    #[test]
    fn object_sha_only_for_embedded_objects() {
        assert_eq!(
            check(|ev| ev.object_sha256 = Some("deadbeef".into())),
            vec![Drift::ObjectSha {
                symbol: "toy_a".into(),
                contract: "abc123".into(),
                evidence: "deadbeef".into()
            }]
        );
        assert_eq!(check(|ev| ev.object_sha256 = Some("ABC123".into())), vec![]);
        assert_eq!(check(|ev| ev.object_sha256 = None), vec![]);
        // toy_b is hipcc-compiled at load: there is no object to pin.
        let e = &TOY.exports[1];
        let mut ev = evidence(&TOY, e);
        ev.object_sha256 = Some("whatever".into());
        assert_eq!(check_export(&TOY, e, &ev), vec![]);
    }

    #[test]
    fn display_is_one_line_per_drift_and_names_the_symbol() {
        let d = check(|ev| {
            ev.symbol = "other".into();
            ev.arch = Some(Arch::Gfx1201);
            ev.args.retain(|a| a.offset != 24);
            ev.args.push(ArgEvidence {
                name: "zz".into(),
                offset: 200,
                size: 8,
                value_kind: "global_buffer".into(),
                access: None,
            });
            arg_mut(ev, 16).name = "mm".into();
            arg_mut(ev, 20).size = 8;
            arg_mut(ev, 28).value_kind = "global_buffer".into();
            arg_mut(ev, 0).access = Some("write_only".into());
            ev.segment_bytes = Some(1);
            ev.segment_align = Some(1);
            ev.max_flat_workgroup_size = Some(1);
            ev.threads = Some(1);
            ev.dynamic_lds = Some(1);
            ev.group_segment_fixed = Some(1);
            ev.wave = Some(1);
            ev.object_sha256 = Some("x".into());
        });
        assert_eq!(d.len(), 16, "{d:#?}");
        for drift in &d {
            let line = drift.to_string();
            assert!(!line.contains('\n'), "{line}");
            assert!(line.starts_with("toy_a:"), "{line}");
        }
    }

    #[test]
    fn drift_and_evidence_serialize() {
        let d = Drift::Wave {
            symbol: "toy_a".into(),
            contract: 32,
            evidence: 64,
        };
        let json = serde_json::to_value(&d).unwrap();
        assert_eq!(json["wave"]["evidence"], 64);
        let ev = evidence(&TOY, &TOY.exports[0]);
        let json = serde_json::to_value(&ev).unwrap();
        assert_eq!(json["source"], "generator");
        assert_eq!(json["arch"], "gfx1100");
    }
}
