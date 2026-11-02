// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! railgun as a consumer of `hipfire-kernel-abi` (PM/railgun bedrock R1).
//!
//! A [`KernelAbiV1`](hipfire_kernel_abi::KernelAbiV1) contract states the
//! explicit kernarg fields, the hidden block, the numeric/extent facts and the
//! launch geometry of one export. railgun consumes it here without building a
//! second kernarg image: [`KernelProgram::encode_kernarg_into`] stays the only
//! full-image encoder, and the contract supplies
//!
//! * [`launch_record`]: the [`LaunchRecord`] of a contract-validated launch
//!   (the contract's explicit image, tail-padded as the HIP `extra` ABI
//!   documents), which [`crate::kernel::author`] turns into a node;
//! * [`check_node`]: the authored node against the contract — symbol, certified
//!   segment, pointer/scalar/hidden layout, bindings against the validated
//!   operand addresses and extents, and the re-encoded kernarg image against
//!   the contract's explicit bytes and derived hidden words;
//! * [`receipt_evidence`]: a receipt kernel record as plain-data
//!   [`ExportEvidence`] for [`hipfire_kernel_abi::check_export`].
//!
//! Nothing here loosens `author`: a node `author` refused (lowered to
//! `HipDirect`) is refused here too, with its reason.

use std::sync::Arc;

use hipfire_kernel_abi as abi;
use railgun_corpus::Receipt;

use crate::kernel::{
    Arg, DeclaredWord, HiddenKind, KernelNode, KernelProgram, LaunchRecord, Lowering, NodeFacts,
};
use crate::Binding;

/// The [`LaunchRecord`] of one contract-validated launch: symbol, grid, block
/// and dynamic LDS from the invocation; `kernarg` is the explicit image
/// tail-padded with zeros to a multiple of 16 (the HIP `extra` ABI, as
/// [`LaunchRecord::kernarg`] documents). The hidden block is not part of it:
/// [`KernelProgram::kernarg_image`] writes the hidden words.
pub fn launch_record(
    v: &abi::ValidatedInvocationV1<'_>,
    facts: Result<Arc<NodeFacts>, String>,
    allocations: Vec<(u32, Result<Binding, String>)>,
    declared: Vec<DeclaredWord>,
) -> LaunchRecord {
    let explicit = v.explicit();
    let mut kernarg = Vec::with_capacity(explicit.len().next_multiple_of(16));
    kernarg.extend_from_slice(explicit);
    kernarg.resize(explicit.len().next_multiple_of(16), 0);
    LaunchRecord {
        symbol: v.export().symbol.to_owned(),
        grid: v.grid(),
        block: v.block(),
        dynamic_lds: v.dynamic_lds(),
        kernarg,
        declared,
        facts,
        allocations,
    }
}

/// One node arg and the explicit bytes it covers.
struct Segment<'a> {
    start: u32,
    len: u32,
    arg: &'a Arg,
}

/// The node's args as contiguous segments of `[0, explicit_bytes)`.
fn segments(node: &KernelNode) -> Result<Vec<Segment<'_>>, String> {
    let mut out = Vec::with_capacity(node.args.len());
    let mut cursor = 0u32;
    for (offset, arg) in &node.args {
        let len = match arg {
            Arg::Const(bytes) => u32::try_from(bytes.len())
                .map_err(|_| format!("constant at +{offset} is {} bytes", bytes.len()))?,
            Arg::Resource { .. } => 8,
            Arg::Word(_) => 4,
        };
        if *offset != cursor {
            return Err(format!(
                "node args are not contiguous: an arg starts at +{offset}, the previous one ends at +{cursor}"
            ));
        }
        cursor = cursor
            .checked_add(len)
            .ok_or_else(|| format!("arg at +{offset} overflows the kernarg offset"))?;
        out.push(Segment {
            start: *offset,
            len,
            arg,
        });
    }
    if cursor != node.explicit_bytes {
        return Err(format!(
            "node args cover {cursor} bytes, its explicit segment is {}",
            node.explicit_bytes
        ));
    }
    Ok(out)
}

fn field_range(f: &abi::FieldV1) -> std::ops::Range<usize> {
    f.offset as usize..f.offset as usize + f.width as usize
}

fn field_bytes<'v>(explicit: &'v [u8], f: &abi::FieldV1) -> Result<&'v [u8], String> {
    explicit.get(field_range(f)).ok_or_else(|| {
        format!(
            "contract field {} (+{}, {} bytes) lies outside the {}-byte explicit image",
            f.name,
            f.offset,
            f.width,
            explicit.len()
        )
    })
}

fn axis_index(axis: abi::Axis) -> u8 {
    match axis {
        abi::Axis::X => 0,
        abi::Axis::Y => 1,
        abi::Axis::Z => 2,
    }
}

/// railgun's hidden-kind classification against the contract's. `Other`
/// carries no spelling on the railgun side; the `value_kind` strings are
/// compared separately.
fn same_hidden_kind(rg: HiddenKind, contract: abi::HiddenKind) -> bool {
    match (rg, contract) {
        (HiddenKind::BlockCount(a), abi::HiddenKind::BlockCount(b))
        | (HiddenKind::GroupSize(a), abi::HiddenKind::GroupSize(b))
        | (HiddenKind::Remainder(a), abi::HiddenKind::Remainder(b))
        | (HiddenKind::GlobalOffset(a), abi::HiddenKind::GlobalOffset(b)) => a == axis_index(b),
        (HiddenKind::GridDims, abi::HiddenKind::GridDims)
        | (HiddenKind::DynamicLdsSize, abi::HiddenKind::DynamicLdsSize)
        | (HiddenKind::Other, abi::HiddenKind::Other(_)) => true,
        _ => false,
    }
}

/// Node `node` of `program` against the contract of the validated invocation
/// `v`:
///
/// * the node's symbol, arch, certified segment (`== abi.segment_bytes`),
///   explicit end, grid, block and dynamic LDS are the contract's and the
///   invocation's;
/// * every contract `Pointer` field is an [`Arg::Resource`] at that offset whose
///   binding `base + interior` is the operand address in `v.explicit()` (a
///   null operand is an all-zero [`Arg::Const`]), and the bound allocation
///   holds the operand's contract extent from that address;
/// * every `Scalar` field is an [`Arg::Const`] with exactly the explicit bytes
///   or an [`Arg::Word`] at that offset;
/// * no [`Arg::Resource`]/[`Arg::Word`] sits at an offset the contract lacks;
/// * every contract hidden word is present with the same kind and width at the
///   same offset, and the node has no other hidden word;
/// * `program.kernarg_image(node, segment, &program.authored_words())` carries
///   `v.explicit()`'s bytes at every explicit field and
///   `v.hidden_expected()`'s values at the hidden offsets.
///
/// A node `author` refused (`Lowering::HipDirect`: no decomposed args) is an
/// error naming the refusal.
pub fn check_node(
    program: &KernelProgram,
    node: usize,
    v: &abi::ValidatedInvocationV1<'_>,
) -> Result<(), String> {
    check_node_inner(program, node, v)
        .map_err(|e| format!("{}: node {node}: {e}", v.export().symbol))
}

fn check_node_inner(
    program: &KernelProgram,
    node: usize,
    v: &abi::ValidatedInvocationV1<'_>,
) -> Result<(), String> {
    let contract = v.abi();
    let export = v.export();
    let n = program
        .nodes
        .get(node)
        .ok_or_else(|| format!("the program has {} nodes", program.nodes.len()))?;
    if let Lowering::HipDirect { reason } = &n.lowering {
        return Err(format!(
            "authoring refused the launch (lowered to HipDirect): {reason}"
        ));
    }
    if n.symbol != export.symbol {
        return Err(format!(
            "node symbol {:?} is not the contract export {:?} ({})",
            n.symbol, export.symbol, contract.id
        ));
    }
    if program.arch != export.arch.name() {
        return Err(format!(
            "program arch {:?} is not the export's {}",
            program.arch,
            export.arch.name()
        ));
    }
    match n.segment_bytes {
        Some(segment) if segment == contract.segment_bytes => {}
        other => {
            return Err(format!(
                "certified kernarg segment {other:?} is not the contract's {} B ({})",
                contract.segment_bytes, contract.id
            ))
        }
    }
    if n.explicit_bytes != contract.explicit_bytes {
        return Err(format!(
            "node explicit segment is {} B, the contract's {} B ({})",
            n.explicit_bytes, contract.explicit_bytes, contract.id
        ));
    }
    if n.grid != v.grid() || n.block != v.block() || n.dynamic_lds != v.dynamic_lds() {
        return Err(format!(
            "node launch grid {:?} block {:?} lds {} differs from the invocation's grid {:?} block {:?} lds {}",
            n.grid,
            n.block,
            n.dynamic_lds,
            v.grid(),
            v.block(),
            v.dynamic_lds()
        ));
    }

    let explicit = v.explicit();
    let segs = segments(n)?;
    let segment_at = |offset: u32| {
        segs.iter()
            .find(|s| s.start <= offset && offset < s.start + s.len)
    };

    // No pointer/word arg at an offset the contract lacks.
    for s in &segs {
        let field = contract.fields.iter().find(|f| f.offset == s.start);
        match (s.arg, field.map(|f| (f, f.kind))) {
            (Arg::Const(_), _) => {}
            (Arg::Resource { .. }, Some((f, abi::FieldKind::Pointer { .. }))) if f.width == 8 => {}
            (Arg::Word(_), Some((f, abi::FieldKind::Scalar { .. }))) if f.width == 4 => {}
            (arg, None) => {
                return Err(format!(
                    "node arg {arg:?} at +{} is at an offset the contract {} has no field at",
                    s.start, contract.id
                ))
            }
            (arg, Some((f, _))) => {
                return Err(format!(
                "node arg {arg:?} at +{} does not match contract field {} (+{}, {} bytes, {:?})",
                s.start, f.name, f.offset, f.width, f.kind
            ))
            }
        }
    }

    for f in contract.fields {
        let want = field_bytes(explicit, f)?;
        let seg = segment_at(f.offset).ok_or_else(|| {
            format!(
                "contract field {} (+{}) is not carried by any node arg",
                f.name, f.offset
            )
        })?;
        match f.kind {
            abi::FieldKind::Pointer { operand } => {
                let address = u64::from_le_bytes(
                    want.try_into()
                        .map_err(|_| format!("pointer field {} is {} bytes", f.name, f.width))?,
                );
                if address == 0 {
                    // A nullable operand: the node holds the zero constant.
                    match seg.arg {
                        Arg::Const(bytes)
                            if bytes
                                .get(f.offset as usize - seg.start as usize..)
                                .and_then(|b| b.get(..want.len()))
                                == Some(want) => {}
                        arg => {
                            return Err(format!(
                            "pointer field {} is null in the invocation; the node carries {arg:?}",
                            f.name
                        ))
                        }
                    }
                    continue;
                }
                let Arg::Resource { resource, interior } = seg.arg else {
                    return Err(format!(
                        "pointer field {} (+{}) is {:#x} in the invocation but the node carries {:?}, not a resource",
                        f.name, f.offset, address, seg.arg
                    ));
                };
                if seg.start != f.offset {
                    return Err(format!(
                        "pointer field {} (+{}) lies inside the resource arg at +{}",
                        f.name, f.offset, seg.start
                    ));
                }
                let binding = program.resources.get(resource.0 as usize).ok_or_else(|| {
                    format!(
                        "pointer field {} names resource {} of {}",
                        f.name,
                        resource.0,
                        program.resources.len()
                    )
                })?;
                let bound = binding.base.checked_add(*interior).ok_or_else(|| {
                    format!("pointer field {}: base + interior overflows", f.name)
                })?;
                if bound != address {
                    return Err(format!(
                        "pointer field {}: the node binds {:#x} (+{:#x}) = {bound:#x}, the invocation's operand is {address:#x}",
                        f.name, binding.base, interior
                    ));
                }
                let op = contract
                    .operand_index(operand)
                    .map(|i| &contract.operands[i])
                    .ok_or_else(|| {
                        format!("field {} names undeclared operand {operand}", f.name)
                    })?;
                if let Some(extent) = op.extent {
                    let need = abi::eval(&extent, contract, export, v.params())
                        .map_err(|e| format!("operand {operand} extent: {e}"))?;
                    let have = binding.bytes.checked_sub(*interior).unwrap_or(0);
                    if need > have {
                        return Err(format!(
                            "operand {operand} needs {need} bytes from {address:#x}, the bound allocation {:#x}+{} holds {have}",
                            binding.base, binding.bytes
                        ));
                    }
                }
            }
            abi::FieldKind::Scalar { .. } => match seg.arg {
                Arg::Const(bytes) => {
                    let at = f.offset as usize - seg.start as usize;
                    if bytes.get(at..at + want.len()) != Some(want) {
                        return Err(format!(
                            "scalar field {} (+{}): the node constant is {:02x?}, the contract's explicit bytes {want:02x?}",
                            f.name,
                            f.offset,
                            bytes.get(at..at + want.len())
                        ));
                    }
                }
                Arg::Word(_) => {
                    // The value is a patched host word; the re-encoded image
                    // below compares its authored value with the contract's.
                    if seg.start != f.offset || f.width != 4 {
                        return Err(format!(
                            "scalar field {} (+{}, {} bytes) is not exactly the word arg at +{}",
                            f.name, f.offset, f.width, seg.start
                        ));
                    }
                }
                Arg::Resource { .. } => {
                    return Err(format!(
                        "scalar field {} (+{}) is carried by a resource arg",
                        f.name, f.offset
                    ))
                }
            },
        }
    }

    // Hidden words: kind, width and offset in both directions.
    for h in contract.hidden {
        let value_kind = h.kind.value_kind();
        let hit = n
            .hidden
            .iter()
            .find(|x| x.offset == h.offset)
            .ok_or_else(|| {
                format!(
                    "contract hidden {value_kind} (+{}) is missing from the node",
                    h.offset
                )
            })?;
        if hit.size != h.width
            || hit.value_kind != value_kind
            || !same_hidden_kind(hit.kind, h.kind)
        {
            return Err(format!(
                "hidden word +{}: node {} ({} bytes, {:?}), contract {value_kind} ({} bytes)",
                h.offset, hit.value_kind, hit.size, hit.kind, h.width
            ));
        }
    }
    for x in &n.hidden {
        if !contract.hidden.iter().any(|h| h.offset == x.offset) {
            return Err(format!(
                "node hidden {} (+{}) is not in the contract",
                x.value_kind, x.offset
            ));
        }
    }

    // The single full-image encoder's output against the contract.
    let image = program.kernarg_image(
        node,
        contract.segment_bytes as usize,
        &program.authored_words(),
    )?;
    for f in contract.fields {
        let want = field_bytes(explicit, f)?;
        if image.get(field_range(f)) != Some(want) {
            return Err(format!(
                "kernarg image differs from the contract at field {} (+{}): image {:02x?}, contract {want:02x?}",
                f.name,
                f.offset,
                image.get(field_range(f))
            ));
        }
    }
    for (h, value) in v.hidden_expected() {
        let width = h.width as usize;
        if width > 8 || (width < 8 && value >> (8 * width) != 0) {
            return Err(format!(
                "hidden {} = {value} does not fit {width} bytes",
                h.kind.value_kind()
            ));
        }
        let at = h.offset as usize;
        let got = image.get(at..at + width);
        if got != Some(&value.to_le_bytes()[..width]) {
            return Err(format!(
                "kernarg image differs from the contract at hidden {} (+{}): image {got:02x?}, expected {value}",
                h.kind.value_kind(),
                h.offset
            ));
        }
    }
    Ok(())
}

/// The receipt's A1 access of a pointer argument in the contract's
/// (AMDGPU `.actual_access`) spelling.
fn access_spelling(mode: &str) -> Result<&'static str, String> {
    match mode {
        "read" => Ok("read_only"),
        "write" => Ok("write_only"),
        "read_write" => Ok("read_write"),
        other => Err(format!("unknown A1 access mode {other:?}")),
    }
}

/// Plain-data [`abi::ExportEvidence`] (source `RailgunReceipt`) for `symbol`
/// from the receipt's railgun section: every explicit and hidden argument with
/// its metadata name, offset, size and `value_kind`, the A1 access of each
/// pointer, the certified kernarg segment, the receipt's arch and the object
/// digest. Facts the receipt does not carry (segment alignment, launch
/// geometry, LDS, wave) are `None` and therefore not compared.
pub fn receipt_evidence(receipt: &Receipt, symbol: &str) -> Result<abi::ExportEvidence, String> {
    let kernel = receipt
        .railgun
        .kernels
        .iter()
        .find(|k| k.symbol == symbol)
        .ok_or_else(|| {
            format!(
                "receipt {} has no railgun record for {symbol}",
                receipt.object_sha256
            )
        })?;
    let arch = abi::Arch::parse(&receipt.arch).ok_or_else(|| {
        format!(
            "receipt {}: arch {:?} is not a contract arch",
            receipt.object_sha256, receipt.arch
        )
    })?;
    let mut args = Vec::with_capacity(kernel.args.len());
    for a in &kernel.args {
        let access = match (a.value_kind.as_str(), a.mode.as_deref()) {
            ("global_buffer", Some(mode)) => Some(
                access_spelling(mode)
                    .map_err(|e| format!("{symbol} {} +{}: {e}", a.name, a.offset))?
                    .to_owned(),
            ),
            _ => None,
        };
        args.push(abi::ArgEvidence {
            name: a.name.clone(),
            offset: a.offset,
            size: a.size,
            value_kind: a.value_kind.clone(),
            access,
        });
    }
    Ok(abi::ExportEvidence {
        source: abi::EvidenceSource::RailgunReceipt,
        arch: Some(arch),
        symbol: kernel.symbol.clone(),
        args,
        segment_bytes: Some(kernel.kernarg_size),
        segment_align: None,
        max_flat_workgroup_size: None,
        threads: None,
        dynamic_lds: None,
        group_segment_fixed: None,
        wave: None,
        object_sha256: Some(receipt.object_sha256.clone()),
    })
}
