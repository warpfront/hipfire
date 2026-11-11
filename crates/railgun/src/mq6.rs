// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Explicit, fixed-shape MQ6 programs authored from the admitted gfx1151
//! lane-row bundle's real corpus receipt. This module performs no file I/O,
//! device submission, recording admission or cache-table qualification.
//!
//! [`Mq6Facts::from_hit`] consumes a hit already verified by
//! [`railgun_corpus::JitCorpus::lookup`]; constructing an unchecked `Hit` is
//! outside that trust boundary. A1 effects come only from the receipt, never
//! from a name-based pointer-effects table. The spans supplied to [`author`]
//! must describe live allocations probed by the runtime. Their leases and
//! unchanged addresses must outlive every execution of the resulting program.
//! M/K/B and grids are constants; changing shape or allocation requires a new
//! authoring. The M1 effects remain allocation-wide, not an A2 span proof.

use std::sync::Arc;

use railgun_corpus::{Hit, KernelRecord, ObjectOrigin};

use crate::kernel::{self, FactStatus, KernelProgram, LaunchRecord, NodeFacts, ReadCache};
use crate::Binding;

pub const MODULE: &str = "qwen4_lane_rows_pm";
pub const SINGLE_SYMBOL: &str = "gemv_mq6g256v2_xbatch_pm";
pub const X4_SYMBOL: &str = "gemv_mq6g256v2_x4_xbatch_pm";
pub const BUNDLE_SHA256: &str = "bc34ee389bd990386bdfc93d94b66752affcd3fbd33bec212a6b4edcaa51a963";
pub const DEVICE_ELF_SHA256: &str =
    "b0a702507aa5420261ac90742e3665403e47636c24e034918507997bb14c53dc";

/// A live tensor view and the entire allocation returned for its address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mq6Span {
    pub address: u64,
    pub allocation: Binding,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mq6Part {
    pub weight: Mq6Span,
    pub output: Mq6Span,
    pub m: u32,
}

/// One fixed-shape launch. Absent x4 slots encode null A/y and M=0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mq6Op {
    Single {
        part: Mq6Part,
        x: Mq6Span,
        k: u32,
        batch: u32,
    },
    X4 {
        parts: [Option<Mq6Part>; 4],
        x: Mq6Span,
        k: u32,
        batch: u32,
    },
}

/// The two target kernels, derived from one verified, canonical bundle hit.
/// Fields stay private so authoring cannot replace their certified effects.
#[derive(Clone, Debug)]
pub struct Mq6Facts {
    single: Arc<NodeFacts>,
    x4: Arc<NodeFacts>,
}

fn require(condition: bool, message: impl Into<String>) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl Mq6Facts {
    /// Validate the narrow MQ6 contract in a hit obtained through the corpus
    /// lookup's receipt-hash/schema/object verification. Raw `.co` receipts
    /// do not certify the embedded container's identity and are refused here.
    pub fn from_hit(hit: &Hit) -> Result<Self, String> {
        let (record, receipt) = (&hit.record, &hit.receipt);
        require(
            record.arch == "gfx1151"
                && receipt.arch == "gfx1151"
                && record.module == MODULE
                && record.origin == ObjectOrigin::PmNative,
            "MQ6 requires the gfx1151 native qwen4_lane_rows_pm corpus record",
        )?;
        require(
            record.artifact_sha256 == BUNDLE_SHA256
                && receipt.object_sha256 == BUNDLE_SHA256
                && receipt.device_elf_sha256 == DEVICE_ELF_SHA256,
            "MQ6 corpus hit does not bind the admitted bundle and device ELF",
        )?;
        require(
            receipt.schema == railgun_corpus::receipt::RECEIPT_SCHEMA
                && receipt.schema_version == railgun_corpus::receipt::RECEIPT_SCHEMA_VERSION
                && receipt.railgun.section_version
                    == railgun_corpus::receipt::RAILGUN_SECTION_VERSION
                && receipt.tier == "D"
                && !receipt.strict
                && digest(&record.receipt_sha256)
                && digest(&receipt.round_trip_proof_sha256),
            "MQ6 requires a verified schema-9 tier-D receipt",
        )?;
        for (id, status) in [("roundtrip", "pass"), ("g8-metadata-differential", "pass")] {
            require(
                receipt.checks.iter().filter(|c| c.id == id).count() == 1
                    && receipt
                        .checks
                        .iter()
                        .any(|c| c.id == id && c.status == status),
                format!("MQ6 receipt has no unique passing {id} check"),
            )?;
        }
        let mut obligations = receipt.obligations.clone();
        obligations.sort();
        require(
            obligations == ["AddressArithmetic", "InBoundsNoAlias"],
            "MQ6 receipt obligations differ from the span/alias contract",
        )?;
        let single = validated_node(hit, SINGLE_SYMBOL, 36, &[0, 8], &[16], &[24, 28, 32])?;
        let x4 = validated_node(
            hit,
            X4_SYMBOL,
            96,
            &[0, 8, 16, 24, 32],
            &[40, 48, 56, 64],
            &[72, 76, 80, 84, 88, 92],
        )?;
        Ok(Self { single, x4 })
    }
}

fn validated_node(
    hit: &Hit,
    symbol: &str,
    segment: u32,
    reads: &[u32],
    writes: &[u32],
    integers: &[u32],
) -> Result<Arc<NodeFacts>, String> {
    require(
        hit.record.symbols.iter().filter(|s| *s == symbol).count() == 1,
        format!("MQ6 corpus record must export {symbol} exactly once"),
    )?;
    let identities: Vec<_> = hit
        .receipt
        .kernels
        .iter()
        .filter(|k| k.symbol == symbol)
        .collect();
    require(
        identities.len() == 1,
        format!("MQ6 receipt must identify {symbol} exactly once"),
    )?;
    let identity = identities[0];
    require(
        digest(&identity.text_sha256)
            && digest(&identity.kd_sha256)
            && digest(&identity.metadata_sha256)
            && identity.text_range[1] > 0,
        format!("{symbol}: missing text/descriptor/metadata identity"),
    )?;
    let records: Vec<_> = hit
        .receipt
        .railgun
        .kernels
        .iter()
        .filter(|k| k.symbol == symbol)
        .collect();
    require(
        records.len() == 1,
        format!("MQ6 receipt must carry {symbol} facts exactly once"),
    )?;
    let record = records[0];
    validate_record(record, segment, reads, writes, integers)?;
    let node = NodeFacts::from_hit(&hit.record, &hit.receipt, symbol)?;
    require(
        node.facts.status == FactStatus::Proven
            && node.facts.explicit_bytes == segment
            && node.facts.segment_bytes == segment
            && node.facts.hidden.is_empty()
            && node.facts.read_cache == ReadCache::VmemOnly,
        format!("{symbol}: derived M1 facts differ from the certified narrow ABI"),
    )?;
    Ok(Arc::new(node))
}

fn validate_record(
    record: &KernelRecord,
    segment: u32,
    reads: &[u32],
    writes: &[u32],
    integers: &[u32],
) -> Result<(), String> {
    let symbol = &record.symbol;
    record.validate_case(BUNDLE_SHA256, DEVICE_ELF_SHA256, "gfx1151")?;
    require(
        record.status == "proven"
            && record.unresolved.is_empty()
            && record.kernarg_size == segment
            && record.read_cache == "vmem_only"
            && !record.dynamic_kernarg_reads
            && !record.taint_overflow
            && record.pm_obligations.is_empty()
            && record.pm_analyze_error.is_none(),
        format!("{symbol}: A1 must be Proven with no unresolved analysis"),
    )?;
    let side = &record.side;
    let implicit = &record.implicit;
    require(
        side.kernarg_segment == 0
            && side.module_global == 0
            && side.abi_packet == 0
            && side.local_flat == 0
            && side.scratch == 0
            && implicit.hidden_loaded.is_empty()
            && implicit.hidden_grid.is_empty()
            && implicit.hidden_grid_live.is_empty()
            && !implicit.loads_block_count
            && !implicit.loads_group_size,
        format!("{symbol}: unexpected side memory or hidden-argument dependency"),
    )?;
    require(
        record.args.len() == reads.len() + writes.len() + integers.len(),
        format!("{symbol}: unexpected explicit/hidden argument count"),
    )?;
    for (offsets, write) in [(reads, false), (writes, true)] {
        for &offset in offsets {
            let args: Vec<_> = record.args.iter().filter(|a| a.offset == offset).collect();
            require(
                args.len() == 1,
                format!("{symbol}: missing/duplicate +{offset}"),
            )?;
            let arg = args[0];
            let (mode, class, cache) = if write {
                ("write", "vmem_store", "not_read")
            } else {
                ("read", "vmem_load", "vmem_only")
            };
            require(
                arg.size == 8
                    && arg.value_kind == "global_buffer"
                    && arg.role == "pointer"
                    && arg.mode.as_deref() == Some(mode)
                    && arg.classes == [class]
                    && arg.read_cache == cache
                    && arg.loaded,
                format!("{symbol} +{offset}: unexpected certified pointer role/access"),
            )?;
        }
    }
    for &offset in integers {
        let args: Vec<_> = record.args.iter().filter(|a| a.offset == offset).collect();
        require(
            args.len() == 1,
            format!("{symbol}: missing/duplicate +{offset}"),
        )?;
        let arg = args[0];
        require(
            arg.size == 4
                && arg.value_kind == "by_value"
                && arg.role == "integer"
                && arg.mode.is_none()
                && arg.classes.is_empty()
                && arg.read_cache == "not_read"
                && arg.loaded,
            format!("{symbol} +{offset}: expected a loaded Integer scalar"),
        )?;
    }
    let differential = &record.differential;
    require(
        differential.source == "object-metadata"
            && differential.status == "pass"
            && differential.args.len() == reads.len() + writes.len(),
        format!("{symbol}: G8 must pass against the object's metadata"),
    )?;
    for (offsets, write) in [(reads, false), (writes, true)] {
        for &offset in offsets {
            let checks: Vec<_> = differential
                .args
                .iter()
                .filter(|a| a.offset == offset)
                .collect();
            require(
                checks.len() == 1,
                format!("{symbol}: missing/duplicate G8 +{offset}"),
            )?;
            let check = checks[0];
            require(
                check.explicit
                    && check.status == "pass"
                    && check.a1.as_deref() == Some(if write { "write" } else { "read" })
                    && check.metadata == if write { "write_only" } else { "read_only" },
                format!("{symbol} +{offset}: G8 access differs from the narrow contract"),
            )?;
        }
    }
    Ok(())
}

fn product(values: &[u64]) -> Result<u64, String> {
    let bytes = values.iter().try_fold(1u64, |a, b| a.checked_mul(*b));
    let bytes = bytes.ok_or("MQ6 span multiplication overflows")?;
    require(
        bytes <= u64::from(u32::MAX),
        "MQ6 span exceeds 32-bit byte offsets",
    )?;
    Ok(bytes)
}

#[derive(Clone, Copy)]
struct Interval {
    begin: u64,
    end: u64,
}

impl Interval {
    fn overlaps(self, other: Self) -> bool {
        self.begin < other.end && other.begin < self.end
    }
}

fn span(
    value: Mq6Span,
    required: u64,
    label: &str,
    allocations: &mut Vec<Binding>,
) -> Result<Interval, String> {
    require(
        value.address % 4 == 0,
        format!("{label}: MQ6 address must be 4-byte aligned"),
    )?;
    let allocation_end = value
        .allocation
        .base
        .checked_add(value.allocation.bytes)
        .ok_or_else(|| format!("{label}: allocation address overflows"))?;
    let view_end = value
        .address
        .checked_add(value.bytes)
        .ok_or_else(|| format!("{label}: view address overflows"))?;
    require(
        value.address != 0
            && value.allocation.base != 0
            && value.allocation.bytes != 0
            && value.address >= value.allocation.base
            && value.address < allocation_end
            && view_end <= allocation_end
            && value.bytes >= required,
        format!("{label}: live view/allocation does not contain the required {required} bytes"),
    )?;
    let whole = Interval {
        begin: value.allocation.base,
        end: allocation_end,
    };
    for other in allocations.iter().copied() {
        require(
            other == value.allocation
                || !whole.overlaps(Interval {
                    begin: other.base,
                    end: other.base + other.bytes,
                }),
            "MQ6 overlapping addresses have inconsistent allocation identities",
        )?;
    }
    if !allocations.contains(&value.allocation) {
        allocations.push(value.allocation);
    }
    Ok(Interval {
        begin: value.address,
        end: value.address + required,
    })
}

fn padding(symbol: &str, bytes: &[u8]) -> Result<(), String> {
    let (explicit, recorded) = match symbol {
        SINGLE_SYMBOL => (36, 48),
        X4_SYMBOL => (96, 96),
        _ => return Err("not a narrow MQ6 symbol".into()),
    };
    require(
        bytes.len() == recorded && bytes[explicit..].iter().all(|b| *b == 0),
        "MQ6 recorded ABI length or zero padding differs from the narrow contract",
    )
}

/// Author a nonempty straight-line sequence. Per-node output/input alias is
/// refused, but a later node may read an earlier node's output; M1 then
/// derives the conservative allocation-wide RAW edge from the receipt.
pub fn author(facts: &Mq6Facts, ops: &[Mq6Op]) -> Result<KernelProgram, String> {
    require(
        !ops.is_empty(),
        "MQ6 authoring needs at least one operation",
    )?;
    let mut launches = Vec::with_capacity(ops.len());
    let mut allocations = Vec::new();
    for op in ops {
        let (parts, x, k, batch, single) = match op {
            Mq6Op::Single { part, x, k, batch } => {
                ([Some(*part), None, None, None], *x, *k, *batch, true)
            }
            Mq6Op::X4 { parts, x, k, batch } => (*parts, *x, *k, *batch, false),
        };
        require(
            k > 0 && k % 256 == 0 && k <= i32::MAX as u32 && (1..=8).contains(&batch),
            "MQ6 needs K > 0 divisible by 256, signed-32-bit K and B in 1..=8",
        )?;
        require(
            parts.iter().any(Option::is_some),
            "MQ6 x4 needs a nonempty matrix",
        )?;
        let mut reads = vec![span(
            x,
            product(&[u64::from(batch), u64::from(k), 4])?,
            "x",
            &mut allocations,
        )?];
        let mut writes = Vec::new();
        let mut grid = 0u32;
        for part in parts.iter().flatten() {
            require(
                part.m > 0 && part.m <= i32::MAX as u32,
                "MQ6 active M must be positive signed-32-bit",
            )?;
            grid = grid
                .checked_add(part.m.div_ceil(4))
                .ok_or("MQ6 workgroup grid overflows")?;
            reads.push(span(
                part.weight,
                product(&[u64::from(part.m), u64::from(k / 256), 200])?,
                "weight",
                &mut allocations,
            )?);
            writes.push(span(
                part.output,
                product(&[u64::from(batch), u64::from(part.m), 4])?,
                "output",
                &mut allocations,
            )?);
        }
        for (index, output) in writes.iter().copied().enumerate() {
            require(
                !reads.iter().any(|read| output.overlaps(*read))
                    && !writes[..index].iter().any(|other| output.overlaps(*other)),
                "MQ6 output aliases an input, weight or another output in the same node",
            )?;
        }
        let mut bytes = Vec::with_capacity(if single { 48 } else { 96 });
        let mut bindings = Vec::new();
        let mut pointer = |value: Option<Mq6Span>| {
            let offset = bytes.len() as u32;
            bytes.extend(value.map_or(0, |s| s.address).to_le_bytes());
            if let Some(value) = value {
                bindings.push((offset, Ok(value.allocation)));
            }
        };
        if single {
            let part = parts[0].expect("single operation has its one matrix");
            pointer(Some(part.weight));
            pointer(Some(x));
            pointer(Some(part.output));
            for value in [part.m, k, batch] {
                bytes.extend(value.to_le_bytes());
            }
            bytes.resize(48, 0);
        } else {
            for part in parts {
                pointer(part.map(|p| p.weight));
            }
            pointer(Some(x));
            for part in parts {
                pointer(part.map(|p| p.output));
            }
            for part in parts {
                bytes.extend(part.map_or(0, |p| p.m).to_le_bytes());
            }
            bytes.extend(k.to_le_bytes());
            bytes.extend(batch.to_le_bytes());
        }
        let node = if single { &facts.single } else { &facts.x4 };
        padding(&node.kref.symbol, &bytes)?;
        launches.push(LaunchRecord {
            symbol: node.kref.symbol.clone(),
            grid: [grid, 1, 1],
            block: [32, 1, 1],
            dynamic_lds: 0,
            kernarg: bytes,
            declared: Vec::new(),
            facts: Ok(node.clone()),
            allocations: bindings,
        });
    }
    let program = kernel::author("gfx1151", &launches);
    require(
        program.nodes.iter().all(|n| n.lowering.is_pm4()),
        "MQ6 facts or bindings did not author exclusively PM4 kernel nodes",
    )?;
    Ok(program)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Root-owned fixture: the complete real bundle receipt and its corpus
    // record. No test constructs successful production A1 facts by hand.
    fn hit() -> Hit {
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/native_mq6_hit.json")).unwrap();
        Hit {
            record: serde_json::from_value(value["record"].take()).unwrap(),
            receipt: serde_json::from_value(value["receipt"].take()).unwrap(),
        }
    }

    fn facts() -> Mq6Facts {
        Mq6Facts::from_hit(&hit()).unwrap()
    }

    // Host-only addresses: these tests never submit them to a device.
    fn view(address: u64, bytes: u64) -> Mq6Span {
        Mq6Span {
            address,
            allocation: Binding {
                base: address,
                bytes,
            },
            bytes,
        }
    }

    fn part(m: u32, k: u32, batch: u32, base: u64) -> Mq6Part {
        Mq6Part {
            weight: view(base, u64::from(m) * u64::from(k / 256) * 200),
            output: view(base + 0x100000, u64::from(batch) * u64::from(m) * 4),
            m,
        }
    }

    #[test]
    fn actual_hit_authors_single_x4_and_same_allocation_raw_chain() {
        let facts = facts();
        let one = Mq6Op::Single {
            part: part(19, 512, 8, 0x100000),
            x: view(0x400000, 8 * 512 * 4),
            k: 512,
            batch: 8,
        };
        let program = author(&facts, &[one]).unwrap();
        assert_eq!(program.nodes[0].grid, [5, 1, 1]);
        assert_eq!(program.nodes[0].block, [32, 1, 1]);
        assert_eq!(program.nodes[0].dynamic_lds, 0);
        assert_eq!(program.nodes[0].explicit_bytes, 36);
        assert_eq!(program.nodes[0].segment_bytes, Some(36));
        assert_eq!(program.kernarg_image(0, 36, &[]).unwrap().len(), 36);
        assert!(program.kernarg_image(0, 48, &[]).is_err());

        let four = Mq6Op::X4 {
            parts: [
                Some(part(19, 512, 8, 0x100000)),
                None,
                Some(part(7, 512, 8, 0x500000)),
                None,
            ],
            x: view(0x900000, 8 * 512 * 4),
            k: 512,
            batch: 8,
        };
        let program = author(&facts, &[four]).unwrap();
        assert_eq!(program.nodes[0].grid, [7, 1, 1]);
        assert_eq!(program.resources.len(), 5);
        let image = program.kernarg_image(0, 96, &[]).unwrap();
        for offset in [8, 24, 48, 64] {
            assert_eq!(&image[offset..offset + 8], &[0; 8]);
        }
        for offset in [76, 84] {
            assert_eq!(&image[offset..offset + 4], &[0; 4]);
        }

        let first = part(256, 256, 3, 0x100000);
        let second = part(256, 256, 3, 0x500000);
        let chain = [
            Mq6Op::Single {
                part: first,
                x: view(0x900000, 3 * 256 * 4),
                k: 256,
                batch: 3,
            },
            Mq6Op::Single {
                part: second,
                x: first.output,
                k: 256,
                batch: 3,
            },
        ];
        let program = author(&facts, &chain).unwrap();
        let written = program.nodes[0]
            .effects
            .iter()
            .find(|e| e.arg_offset == 16)
            .unwrap();
        let read = program.nodes[1]
            .effects
            .iter()
            .find(|e| e.arg_offset == 8)
            .unwrap();
        assert!(written.writes() && read.reads());
        assert_eq!(written.resource, read.resource);
        assert!(program.nodes.iter().all(|n| n.lowering.is_pm4()));
    }

    #[test]
    fn refuses_shapes_and_32_bit_span_overflow() {
        let facts = facts();
        let p = part(19, 512, 8, 0x100000);
        let x = view(0x400000, 8 * 512 * 4);
        for (k, batch) in [(0, 1), (255, 1), (512, 0), (512, 9), (1 << 31, 1)] {
            assert!(author(
                &facts,
                &[Mq6Op::Single {
                    part: p,
                    x,
                    k,
                    batch
                }]
            )
            .is_err());
        }
        for m in [0, u32::MAX, i32::MAX as u32] {
            let mut p = p;
            p.m = m;
            assert!(author(
                &facts,
                &[Mq6Op::Single {
                    part: p,
                    x,
                    k: 512,
                    batch: 1
                }]
            )
            .is_err());
        }
        assert!(author(&facts, &[]).is_err());
        assert!(author(
            &facts,
            &[Mq6Op::X4 {
                parts: [None; 4],
                x,
                k: 512,
                batch: 1
            }]
        )
        .is_err());
    }

    #[test]
    fn refuses_short_views_address_overflow_and_same_node_aliases() {
        let facts = facts();
        let p = part(19, 512, 8, 0x100000);
        let x = view(0x400000, 8 * 512 * 4);
        let run = |part, x| {
            author(
                &facts,
                &[Mq6Op::Single {
                    part,
                    x,
                    k: 512,
                    batch: 8,
                }],
            )
        };
        // Preserve both nominal view size and allocation bounds so this
        // negative can fail only the dword-alignment contract.
        let mut unaligned = x;
        unaligned.address += 1;
        unaligned.allocation.bytes += 4;
        assert!(unaligned.address >= unaligned.allocation.base);
        assert!(
            unaligned.address + unaligned.bytes
                <= unaligned.allocation.base + unaligned.allocation.bytes
        );
        assert!(run(p, unaligned).unwrap_err().contains("4-byte aligned"));
        let mut short = x;
        short.bytes -= 1;
        assert!(run(p, short).is_err());
        let mut beyond = x;
        beyond.allocation.bytes -= 1;
        assert!(run(p, beyond).is_err());
        let mut overflow = x;
        overflow.allocation.base = u64::MAX - 7;
        overflow.allocation.bytes = 16;
        assert!(run(p, overflow).is_err());
        let mut overflow = x;
        overflow.address = u64::MAX - 7;
        overflow.bytes = 16;
        assert!(run(p, overflow).is_err());
        let mut alias = p;
        alias.output = x;
        assert!(run(alias, x).is_err());
        let mut alias = p;
        alias.output = p.weight;
        assert!(run(alias, x).is_err());

        let mut other = part(7, 512, 8, 0x600000);
        other.output = p.output;
        assert!(author(
            &facts,
            &[Mq6Op::X4 {
                parts: [Some(p), None, Some(other), None],
                x,
                k: 512,
                batch: 8,
            }]
        )
        .is_err());

        // Distinct allocation identities cannot hide overlapping storage
        // from M1's allocation-wide resource deduplication.
        let mut conflicting = x;
        conflicting.allocation.base = p.weight.address - 16;
        conflicting.allocation.bytes = x.bytes + 32;
        conflicting.address = p.weight.address;
        assert!(run(p, conflicting).is_err());
    }

    #[test]
    fn recorded_padding_is_zero_and_distinct_from_loader_segment() {
        let mut single = vec![0u8; 48];
        assert!(padding(SINGLE_SYMBOL, &single).is_ok());
        single[36] = 1;
        assert!(padding(SINGLE_SYMBOL, &single).is_err());
        assert!(padding(SINGLE_SYMBOL, &[0; 36]).is_err());
        assert!(padding(X4_SYMBOL, &[0; 96]).is_ok());
        assert!(padding(X4_SYMBOL, &[0; 112]).is_err());
    }

    #[test]
    fn actual_receipt_mutations_are_refused_instead_of_inventing_facts() {
        let mut value = hit();
        value.receipt.schema_version = 7;
        assert!(Mq6Facts::from_hit(&value).is_err());
        let mut value = hit();
        value.receipt.railgun.kernels[0]
            .guards
            .push(railgun_corpus::receipt::GuardRecord {
                kind: "kernarg_eq".into(),
                arg_offset: 0,
                size: 8,
                value: "0".into(),
            });
        assert!(Mq6Facts::from_hit(&value).is_err());
        let mut value = hit();
        value.receipt.railgun.kernels[0].status = "unknown".into();
        assert!(Mq6Facts::from_hit(&value).is_err());
        let mut value = hit();
        value.receipt.railgun.kernels[0].differential.status = "fail".into();
        assert!(Mq6Facts::from_hit(&value).is_err());
        let mut value = hit();
        value.receipt.railgun.kernels[0].args[0].classes = vec!["smem_load".into()];
        assert!(Mq6Facts::from_hit(&value).is_err());
        let mut value = hit();
        value.receipt.railgun.kernels[0].args[0].offset = 1;
        assert!(Mq6Facts::from_hit(&value).is_err());
        let mut value = hit();
        value.receipt.railgun.kernels[0].args[3].role = "pointer".into();
        assert!(Mq6Facts::from_hit(&value).is_err());
        let mut value = hit();
        value.receipt.railgun.kernels[0]
            .implicit
            .hidden_loaded
            .push("hidden_queue_ptr".into());
        assert!(Mq6Facts::from_hit(&value).is_err());
        let mut value = hit();
        value.receipt.device_elf_sha256 = "0".repeat(64);
        assert!(Mq6Facts::from_hit(&value).is_err());
        let mut value = hit();
        value.record.artifact_sha256 = DEVICE_ELF_SHA256.into();
        value.receipt.object_sha256 = DEVICE_ELF_SHA256.into();
        assert!(Mq6Facts::from_hit(&value).is_err());
        let mut value = hit();
        value.record.origin = ObjectOrigin::Hipcc;
        assert!(Mq6Facts::from_hit(&value).is_err());
    }
}
