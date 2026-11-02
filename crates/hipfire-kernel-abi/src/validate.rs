// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Invocation validation, explicit-image encoding/decoding and the internal
//! consistency check of a [`KernelAbiV1`].
//!
//! The success path of [`validate_invocation`], [`encode_into`] and
//! [`ValidatedInvocationV1::kernel_params`] performs no heap allocation: the
//! explicit image and the params live in fixed arrays inside the returned
//! value. Only [`AbiError`] construction on a failure path allocates.

use core::ffi::c_void;

use crate::expr::{eval, visit};
use crate::{
    AbiError, Axis, ExportV1, Expr, FieldKind, FieldV1, HiddenKind, HiddenV1, InvocationV1,
    KernelAbiV1, OperandV1, Rule, ScalarTy,
};

/// Largest explicit kernarg image a contract may declare.
pub const MAX_EXPLICIT_BYTES: usize = 512;
/// Largest number of params / explicit fields a contract may declare (`raon_rda_conv_prepare_w128_f32`
/// takes 33 kernel arguments).
pub const MAX_FIELDS: usize = 40;
/// Operands are tracked in a `u64` bitmask.
const MAX_OPERANDS: usize = 64;
/// Per-workgroup LDS limit of every supported arch.
const MAX_LDS_BYTES: u64 = 65536;
/// Largest flat workgroup size.
const MAX_BLOCK_THREADS: u64 = 1024;

fn scalar_bytes(ty: ScalarTy) -> u32 {
    match ty {
        ScalarTy::I32 | ScalarTy::U32 => 4,
        ScalarTy::U64 => 8,
    }
}

fn fits(ty: ScalarTy, v: u64) -> bool {
    match ty {
        ScalarTy::I32 => v <= i32::MAX as u64,
        ScalarTy::U32 => v <= u64::from(u32::MAX),
        ScalarTy::U64 => true,
    }
}

/// Product of the block dimensions, saturating.
pub(crate) fn block_product(block: [u32; 3]) -> u64 {
    block
        .iter()
        .fold(1u64, |acc, &b| acc.saturating_mul(u64::from(b)))
}

fn contract_err(abi: &KernelAbiV1, reason: String) -> AbiError {
    AbiError::Contract {
        abi: abi.id,
        reason,
    }
}

fn expr_err(abi: &KernelAbiV1, expr: &Expr, reason: String) -> AbiError {
    AbiError::Expr {
        abi: abi.id,
        expr: format!("{expr:?}"),
        reason,
    }
}

fn widen(a: [u32; 3]) -> [u64; 3] {
    [u64::from(a[0]), u64::from(a[1]), u64::from(a[2])]
}

fn axis_index(a: Axis) -> usize {
    match a {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

fn check_limits(abi: &KernelAbiV1) -> Result<(), AbiError> {
    if abi.explicit_bytes as usize > MAX_EXPLICIT_BYTES {
        return Err(contract_err(
            abi,
            format!(
                "explicit_bytes {} exceeds MAX_EXPLICIT_BYTES {MAX_EXPLICIT_BYTES}",
                abi.explicit_bytes
            ),
        ));
    }
    if abi.params.len() > MAX_FIELDS {
        return Err(contract_err(
            abi,
            format!("{} params exceed MAX_FIELDS {MAX_FIELDS}", abi.params.len()),
        ));
    }
    if abi.fields.len() > MAX_FIELDS {
        return Err(contract_err(
            abi,
            format!(
                "{} explicit fields exceed MAX_FIELDS {MAX_FIELDS}",
                abi.fields.len()
            ),
        ));
    }
    if abi.operands.len() > MAX_OPERANDS {
        return Err(contract_err(
            abi,
            format!(
                "{} operands exceed the {MAX_OPERANDS}-operand limit",
                abi.operands.len()
            ),
        ));
    }
    Ok(())
}

/// A launch that passed [`validate_invocation`], with its explicit kernarg
/// image already encoded.
#[derive(Clone, Debug)]
pub struct ValidatedInvocationV1<'a> {
    abi: &'a KernelAbiV1,
    export: &'a ExportV1,
    explicit: [u8; MAX_EXPLICIT_BYTES],
    params: [u64; MAX_FIELDS],
    params_len: usize,
    grid: [u32; 3],
    block: [u32; 3],
    dynamic_lds: u32,
    /// Bit `i` set: operand `i` was bound without a byte count.
    unchecked: u64,
}

impl<'a> ValidatedInvocationV1<'a> {
    pub fn abi(&self) -> &'a KernelAbiV1 {
        self.abi
    }

    pub fn export(&self) -> &'a ExportV1 {
        self.export
    }

    /// The explicit kernarg image, exactly `abi.explicit_bytes` long. Bytes
    /// that no explicit field covers (a hidden region inside it, padding)
    /// are zero.
    pub fn explicit(&self) -> &[u8] {
        &self.explicit[..self.abi.explicit_bytes as usize]
    }

    /// The validated params in [`KernelAbiV1::params`] order.
    pub fn params(&self) -> &[u64] {
        &self.params[..self.params_len]
    }

    pub fn param(&self, name: &str) -> Option<u64> {
        self.abi.param_index(name).map(|i| self.params[i])
    }

    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }

    pub fn block(&self) -> [u32; 3] {
        self.block
    }

    pub fn dynamic_lds(&self) -> u32 {
        self.dynamic_lds
    }

    /// Operands whose byte extent was not proven because the caller bound
    /// them without a byte count.
    pub fn unchecked_extents(&self) -> impl Iterator<Item = &'a OperandV1> + '_ {
        let mask = self.unchecked;
        let operands: &'a [OperandV1] = self.abi.operands;
        operands
            .iter()
            .enumerate()
            .filter(move |(i, _)| (mask >> *i) & 1 == 1)
            .map(|(_, o)| o)
    }

    /// One pointer per explicit field, in field (offset) order, into this
    /// value's explicit image: the `kernelParams` array of an ordinary HIP
    /// launch. The pointers are valid while `self` is neither moved nor
    /// dropped. Returns the number of entries written.
    pub fn kernel_params(&mut self, out: &mut [*mut c_void]) -> Result<usize, AbiError> {
        let n = self.abi.fields.len();
        if out.len() < n {
            let word = core::mem::size_of::<*mut c_void>();
            return Err(AbiError::Buffer {
                abi: self.abi.id,
                need: n * word,
                got: out.len() * word,
            });
        }
        let base = self.explicit.as_mut_ptr();
        for (slot, f) in out.iter_mut().zip(self.abi.fields.iter()) {
            *slot = base.wrapping_add(f.offset as usize).cast::<c_void>();
        }
        Ok(n)
    }

    /// `(hidden word, value)` for every hidden kind derivable from the
    /// launch. `HiddenKind::Other` words are skipped: their value is the
    /// runtime's.
    pub fn hidden_expected(&self) -> impl Iterator<Item = (HiddenV1, u64)> + '_ {
        let (grid, block, lds) = (self.grid, self.block, self.dynamic_lds);
        let dims: u64 = if grid[2] != 1 || block[2] != 1 {
            3
        } else if grid[1] != 1 || block[1] != 1 {
            2
        } else {
            1
        };
        let hidden: &'static [HiddenV1] = self.abi.hidden;
        hidden.iter().filter_map(move |h| {
            let value = match h.kind {
                HiddenKind::BlockCount(a) => u64::from(grid[axis_index(a)]),
                HiddenKind::GroupSize(a) => u64::from(block[axis_index(a)]),
                HiddenKind::Remainder(_) | HiddenKind::GlobalOffset(_) => 0,
                HiddenKind::GridDims => dims,
                HiddenKind::DynamicLdsSize => u64::from(lds),
                HiddenKind::Other(_) => return None,
            };
            Some((*h, value))
        })
    }
}

fn rule_holds(
    rule: &Rule,
    abi: &KernelAbiV1,
    export: &ExportV1,
    params: &[u64],
) -> Result<bool, AbiError> {
    let ev = |e: &Expr| eval(e, abi, export, params).map_err(|reason| expr_err(abi, e, reason));
    Ok(match rule {
        Rule::Positive(e) => ev(e)? > 0,
        Rule::MultipleOf(e, m) => {
            if *m == 0 {
                return Err(contract_err(
                    abi,
                    format!("domain rule {rule:?} is a multiple of zero"),
                ));
            }
            ev(e)? % m == 0
        }
        Rule::AtMost(e, limit) => ev(e)? <= *limit,
        Rule::AtLeast(e, limit) => ev(e)? >= *limit,
        Rule::Equal(a, b) => ev(a)? == ev(b)?,
        Rule::OneOf(e, set) => set.contains(&ev(e)?),
    })
}

fn encode_fields(
    abi: &KernelAbiV1,
    export: &ExportV1,
    params: &[u64],
    operands: &[crate::OperandBinding],
    image: &mut [u8; MAX_EXPLICIT_BYTES],
) -> Result<(), AbiError> {
    let explicit_bytes = abi.explicit_bytes as usize;
    for f in abi.fields.iter() {
        let start = f.offset as usize;
        let width = f.width as usize;
        let end = start.checked_add(width).filter(|&e| e <= explicit_bytes);
        let Some(end) = end else {
            return Err(contract_err(
                abi,
                format!(
                    "field {} [{start}, {start}+{width}) lies outside the {explicit_bytes}-byte explicit image",
                    f.name
                ),
            ));
        };
        match f.kind {
            FieldKind::Pointer { operand } => {
                if width != 8 {
                    return Err(contract_err(
                        abi,
                        format!("pointer field {} has width {width}, expected 8", f.name),
                    ));
                }
                let idx = abi.operand_index(operand).ok_or_else(|| {
                    contract_err(
                        abi,
                        format!("pointer field {} names unknown operand {operand}", f.name),
                    )
                })?;
                image[start..end].copy_from_slice(&operands[idx].address.to_le_bytes());
            }
            FieldKind::Scalar { ty, value } => {
                if width != scalar_bytes(ty) as usize {
                    return Err(contract_err(
                        abi,
                        format!(
                            "scalar field {} has width {width}, {ty:?} needs {}",
                            f.name,
                            scalar_bytes(ty)
                        ),
                    ));
                }
                let v = eval(&value, abi, export, params)
                    .map_err(|reason| expr_err(abi, &value, reason))?;
                if !fits(ty, v) {
                    return Err(AbiError::ParamRange {
                        abi: abi.id,
                        name: f.name,
                        value: v,
                        ty,
                    });
                }
                match ty {
                    ScalarTy::U64 => image[start..end].copy_from_slice(&v.to_le_bytes()),
                    ScalarTy::I32 | ScalarTy::U32 => {
                        image[start..end].copy_from_slice(&(v as u32).to_le_bytes())
                    }
                }
            }
        }
    }
    Ok(())
}

/// Validate `inv` against `abi` and encode its explicit image.
///
/// Order: export lookup by `(arch, symbol)` → param/operand counts → param
/// ranges → domain rules → block, dynamic LDS, grid → per operand (format,
/// null, alignment, extent) → explicit image. The first failure is returned.
/// A null operand that the contract admits (`nullable`) skips the alignment
/// and extent checks: the kernel never dereferences it.
pub fn validate_invocation<'a>(
    abi: &'a KernelAbiV1,
    inv: &InvocationV1<'_>,
) -> Result<ValidatedInvocationV1<'a>, AbiError> {
    let export = abi
        .export(inv.arch, inv.symbol)
        .ok_or_else(|| AbiError::UnknownExport {
            abi: abi.id,
            arch: inv.arch.name(),
            symbol: inv.symbol.to_string(),
        })?;
    check_limits(abi)?;

    if inv.params.len() != abi.params.len() {
        return Err(AbiError::ParamCount {
            abi: abi.id,
            count: inv.params.len(),
            expected: abi.params.len(),
        });
    }
    if inv.operands.len() != abi.operands.len() {
        return Err(AbiError::OperandCount {
            abi: abi.id,
            count: inv.operands.len(),
            expected: abi.operands.len(),
        });
    }

    for (p, &v) in abi.params.iter().zip(inv.params) {
        if !fits(p.ty, v) {
            return Err(AbiError::ParamRange {
                abi: abi.id,
                name: p.name,
                value: v,
                ty: p.ty,
            });
        }
    }

    for d in abi.domain.iter() {
        if !rule_holds(&d.rule, abi, export, inv.params)? {
            return Err(AbiError::Domain {
                abi: abi.id,
                rule: d.rule,
                why: d.why,
            });
        }
    }

    if inv.block != export.block {
        return Err(AbiError::Launch {
            abi: abi.id,
            what: "block",
            expected: widen(export.block),
            got: widen(inv.block),
        });
    }
    if inv.dynamic_lds != export.dynamic_lds {
        return Err(AbiError::Launch {
            abi: abi.id,
            what: "dynamic_lds",
            expected: [u64::from(export.dynamic_lds), 0, 0],
            got: [u64::from(inv.dynamic_lds), 0, 0],
        });
    }
    let mut want_grid = [0u64; 3];
    for (slot, e) in want_grid.iter_mut().zip(export.grid.iter()) {
        *slot = eval(e, abi, export, inv.params).map_err(|reason| expr_err(abi, e, reason))?;
    }
    if want_grid != widen(inv.grid) {
        return Err(AbiError::Launch {
            abi: abi.id,
            what: "grid",
            expected: want_grid,
            got: widen(inv.grid),
        });
    }

    let mut unchecked = 0u64;
    for (i, (op, bind)) in abi.operands.iter().zip(inv.operands).enumerate() {
        if bind.format != op.format {
            return Err(AbiError::Format {
                abi: abi.id,
                operand: op.name,
                expected: op.format,
                got: bind.format,
            });
        }
        if bind.address == 0 {
            if op.nullable {
                continue;
            }
            return Err(AbiError::Null {
                abi: abi.id,
                operand: op.name,
            });
        }
        let align = u64::from(op.align);
        if align == 0 {
            return Err(contract_err(
                abi,
                format!("operand {} declares a zero alignment", op.name),
            ));
        }
        if bind.address % align != 0 {
            return Err(AbiError::Misaligned {
                abi: abi.id,
                operand: op.name,
                address: bind.address,
                align: op.align,
            });
        }
        match (op.extent, bind.bytes) {
            (Some(extent), Some(have)) => {
                let need = eval(&extent, abi, export, inv.params)
                    .map_err(|reason| expr_err(abi, &extent, reason))?;
                if need > have {
                    return Err(AbiError::Extent {
                        abi: abi.id,
                        operand: op.name,
                        need,
                        have,
                    });
                }
            }
            (_, None) => unchecked |= 1u64 << i,
            (None, Some(_)) => {}
        }
    }

    let mut explicit = [0u8; MAX_EXPLICIT_BYTES];
    encode_fields(abi, export, inv.params, inv.operands, &mut explicit)?;

    let mut params = [0u64; MAX_FIELDS];
    params[..inv.params.len()].copy_from_slice(inv.params);

    Ok(ValidatedInvocationV1 {
        abi,
        export,
        explicit,
        params,
        params_len: inv.params.len(),
        grid: inv.grid,
        block: inv.block,
        dynamic_lds: inv.dynamic_lds,
        unchecked,
    })
}

/// Copy the explicit image into `out` and zero the rest of `out`.
/// `out.len()` must be at least `abi.explicit_bytes`.
pub fn encode_into(v: &ValidatedInvocationV1<'_>, out: &mut [u8]) -> Result<(), AbiError> {
    let image = v.explicit();
    if out.len() < image.len() {
        return Err(AbiError::Buffer {
            abi: v.abi.id,
            need: image.len(),
            got: out.len(),
        });
    }
    let (head, tail) = out.split_at_mut(image.len());
    head.copy_from_slice(image);
    tail.fill(0);
    Ok(())
}

/// What [`decode_params`] recovered from an explicit image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedV1 {
    /// In [`KernelAbiV1::params`] order; `None` where no field carries the
    /// param directly.
    pub params: Vec<Option<u64>>,
    /// In [`KernelAbiV1::operands`] order: the address each pointer field
    /// holds (0 for an operand no pointer field carries).
    pub addresses: Vec<u64>,
}

fn field_raw(abi: &KernelAbiV1, explicit: &[u8], f: &FieldV1) -> Result<u64, AbiError> {
    let start = f.offset as usize;
    let width = f.width as usize;
    let limit = (abi.explicit_bytes as usize).min(explicit.len());
    let Some(end) = start.checked_add(width).filter(|&e| e <= limit) else {
        return Err(contract_err(
            abi,
            format!(
                "field {} [{start}, {start}+{width}) lies outside the {limit}-byte explicit image",
                f.name
            ),
        ));
    };
    let want = match f.kind {
        FieldKind::Pointer { .. } => 8,
        FieldKind::Scalar { ty, .. } => scalar_bytes(ty) as usize,
    };
    if width != want {
        return Err(contract_err(
            abi,
            format!("field {} has width {width}, its kind needs {want}", f.name),
        ));
    }
    let raw = explicit[start..end]
        .iter()
        .rev()
        .fold(0u64, |acc, &b| (acc << 8) | u64::from(b));
    if let FieldKind::Scalar { ty, .. } = f.kind {
        if !fits(ty, raw) {
            return Err(AbiError::ParamRange {
                abi: abi.id,
                name: f.name,
                value: raw,
                ty,
            });
        }
    }
    Ok(raw)
}

/// Recover params and operand addresses from an explicit image.
///
/// A scalar field whose value is exactly `Expr::Param(p)` of a declared param
/// reports `p` (two fields disagreeing on `p` is an error). Every other
/// scalar field is re-evaluated from the recovered params and must equal the
/// bytes it holds, unless its expression names an export constant (then the
/// export is unknown here) or a param no field carries; such fields are
/// skipped. A reserved `Const(0)` word is therefore checked to be zero.
pub fn decode_params(abi: &KernelAbiV1, explicit: &[u8]) -> Result<DecodedV1, AbiError> {
    let need = abi.explicit_bytes as usize;
    if explicit.len() < need {
        return Err(AbiError::Buffer {
            abi: abi.id,
            need,
            got: explicit.len(),
        });
    }
    let mut params: Vec<Option<u64>> = vec![None; abi.params.len()];
    let mut addresses = vec![0u64; abi.operands.len()];

    for f in abi.fields.iter() {
        let raw = field_raw(abi, explicit, f)?;
        match f.kind {
            FieldKind::Pointer { operand } => {
                let idx = abi.operand_index(operand).ok_or_else(|| {
                    contract_err(
                        abi,
                        format!("pointer field {} names unknown operand {operand}", f.name),
                    )
                })?;
                addresses[idx] = raw;
            }
            FieldKind::Scalar {
                value: Expr::Param(name),
                ..
            } => {
                if let Some(i) = abi.param_index(name) {
                    let earlier = params[i];
                    match earlier {
                        Some(prev) if prev != raw => {
                            return Err(AbiError::Expr {
                                abi: abi.id,
                                expr: format!("Param({name:?})"),
                                reason: format!(
                                    "conflicting values: field {} holds {raw}, an earlier field holds {prev}",
                                    f.name
                                ),
                            });
                        }
                        _ => params[i] = Some(raw),
                    }
                }
            }
            FieldKind::Scalar { .. } => {}
        }
    }

    let values: Vec<u64> = params.iter().map(|p| p.unwrap_or(0)).collect();
    for f in abi.fields.iter() {
        let FieldKind::Scalar { value, .. } = f.kind else {
            continue;
        };
        if matches!(value, Expr::Param(n) if abi.param_index(n).is_some()) {
            continue;
        }
        let mut evaluable = true;
        visit(&value, &mut |e| {
            if let Expr::Param(n) = e {
                match abi.param_index(n) {
                    Some(i) if params[i].is_some() => {}
                    _ => evaluable = false,
                }
            }
        });
        if !evaluable {
            continue;
        }
        let export = abi
            .exports
            .first()
            .ok_or_else(|| contract_err(abi, "contract declares no exports".into()))?;
        let want =
            eval(&value, abi, export, &values).map_err(|reason| expr_err(abi, &value, reason))?;
        let raw = field_raw(abi, explicit, f)?;
        if raw != want {
            return Err(AbiError::Expr {
                abi: abi.id,
                expr: format!("{value:?}"),
                reason: format!(
                    "field {} holds {raw}, the recovered params give {want}",
                    f.name
                ),
            });
        }
    }

    Ok(DecodedV1 { params, addresses })
}

struct Region {
    offset: u32,
    width: u32,
    label: String,
}

impl Region {
    fn end(&self) -> u64 {
        u64::from(self.offset) + u64::from(self.width)
    }
}

/// Internal consistency of `abi`, independent of any evidence. Every problem
/// found is returned, not just the first.
pub(crate) fn self_check(abi: &KernelAbiV1) -> Result<(), Vec<AbiError>> {
    let mut errs: Vec<String> = Vec::new();

    if abi.exports.is_empty() {
        errs.push("declares no exports".into());
    }
    if abi.explicit_bytes as usize > MAX_EXPLICIT_BYTES {
        errs.push(format!(
            "explicit_bytes {} exceeds MAX_EXPLICIT_BYTES {MAX_EXPLICIT_BYTES}",
            abi.explicit_bytes
        ));
    }
    if abi.params.len() > MAX_FIELDS {
        errs.push(format!(
            "{} params exceed MAX_FIELDS {MAX_FIELDS}",
            abi.params.len()
        ));
    }
    if abi.fields.len() > MAX_FIELDS {
        errs.push(format!(
            "{} explicit fields exceed MAX_FIELDS {MAX_FIELDS}",
            abi.fields.len()
        ));
    }
    if abi.operands.len() > MAX_OPERANDS {
        errs.push(format!(
            "{} operands exceed the {MAX_OPERANDS}-operand limit",
            abi.operands.len()
        ));
    }
    if !abi.segment_align.is_power_of_two() {
        errs.push(format!(
            "segment_align {} is not a power of two",
            abi.segment_align
        ));
    }
    if abi.wave == 0 {
        errs.push("wave is zero".into());
    }

    // Names.
    for (i, p) in abi.params.iter().enumerate() {
        if abi.params[..i].iter().any(|q| q.name == p.name) {
            errs.push(format!("param {} is declared twice", p.name));
        }
    }
    for (i, o) in abi.operands.iter().enumerate() {
        if abi.operands[..i].iter().any(|q| q.name == o.name) {
            errs.push(format!("operand {} is declared twice", o.name));
        }
        if !o.align.is_power_of_two() {
            errs.push(format!(
                "operand {} alignment {} is not a power of two",
                o.name, o.align
            ));
        }
    }

    // Layout: fields and hidden words.
    let mut regions: Vec<Region> = Vec::new();
    for f in abi.fields.iter() {
        regions.push(Region {
            offset: f.offset,
            width: f.width,
            label: format!("field {}", f.name),
        });
    }
    for h in abi.hidden.iter() {
        regions.push(Region {
            offset: h.offset,
            width: h.width,
            label: format!("hidden {}", h.kind.value_kind()),
        });
    }
    for w in abi.fields.windows(2) {
        if w[1].offset <= w[0].offset {
            errs.push(format!(
                "fields are not sorted by offset: {} @{} follows {} @{}",
                w[1].name, w[1].offset, w[0].name, w[0].offset
            ));
        }
    }
    for w in abi.hidden.windows(2) {
        if w[1].offset <= w[0].offset {
            errs.push(format!(
                "hidden words are not sorted by offset: {} @{} follows {} @{}",
                w[1].kind.value_kind(),
                w[1].offset,
                w[0].kind.value_kind(),
                w[0].offset
            ));
        }
    }
    for r in &regions {
        if r.width == 0 {
            errs.push(format!("{} has zero width", r.label));
            continue;
        }
        if r.offset % r.width != 0 {
            errs.push(format!(
                "{} at offset {} is not naturally aligned to its width {}",
                r.label, r.offset, r.width
            ));
        }
        if r.end() > u64::from(abi.segment_bytes) {
            errs.push(format!(
                "{} [{}, {}) lies outside the {}-byte kernarg segment",
                r.label,
                r.offset,
                r.end(),
                abi.segment_bytes
            ));
        }
    }
    let mut sorted: Vec<&Region> = regions.iter().collect();
    sorted.sort_by_key(|r| (r.offset, r.width));
    for w in sorted.windows(2) {
        if w[0].end() > u64::from(w[1].offset) {
            errs.push(format!(
                "{} [{}, {}) overlaps {} [{}, {})",
                w[0].label,
                w[0].offset,
                w[0].end(),
                w[1].label,
                w[1].offset,
                w[1].end()
            ));
        }
    }
    let explicit_end = abi
        .fields
        .iter()
        .map(|f| u64::from(f.offset) + u64::from(f.width))
        .max()
        .unwrap_or(0);
    if u64::from(abi.explicit_bytes) != explicit_end {
        errs.push(format!(
            "explicit_bytes is {}, the last explicit field ends at {explicit_end}",
            abi.explicit_bytes
        ));
    }

    // Field kinds and operand carriage.
    let mut carried = vec![0usize; abi.operands.len()];
    for f in abi.fields.iter() {
        match f.kind {
            FieldKind::Pointer { operand } => {
                if f.width != 8 {
                    errs.push(format!(
                        "pointer field {} has width {}, expected 8",
                        f.name, f.width
                    ));
                }
                match abi.operand_index(operand) {
                    Some(i) => carried[i] += 1,
                    None => errs.push(format!(
                        "pointer field {} names unknown operand {operand}",
                        f.name
                    )),
                }
            }
            FieldKind::Scalar { ty, .. } => {
                if f.width != scalar_bytes(ty) {
                    errs.push(format!(
                        "scalar field {} has width {}, {ty:?} needs {}",
                        f.name,
                        f.width,
                        scalar_bytes(ty)
                    ));
                }
            }
        }
    }
    for (o, &n) in abi.operands.iter().zip(&carried) {
        match n {
            1 => {}
            0 => errs.push(format!(
                "operand {} is not carried by any pointer field",
                o.name
            )),
            n => errs.push(format!(
                "operand {} is carried by {n} pointer fields, expected exactly one",
                o.name
            )),
        }
    }

    // Expressions: every name resolves, no zero divisors.
    let mut exprs: Vec<(String, Expr)> = Vec::new();
    for f in abi.fields.iter() {
        if let FieldKind::Scalar { value, .. } = f.kind {
            exprs.push((format!("field {}", f.name), value));
        }
    }
    for o in abi.operands.iter() {
        if let Some(extent) = o.extent {
            exprs.push((format!("operand {} extent", o.name), extent));
        }
    }
    for d in abi.domain.iter() {
        let ctx = format!("domain rule {:?}", d.rule);
        match d.rule {
            Rule::Positive(e) | Rule::AtMost(e, _) | Rule::AtLeast(e, _) | Rule::OneOf(e, _) => {
                exprs.push((ctx, e))
            }
            Rule::MultipleOf(e, m) => {
                if m == 0 {
                    errs.push(format!("{ctx} is a multiple of zero"));
                }
                exprs.push((ctx, e));
            }
            Rule::Equal(a, b) => {
                exprs.push((ctx.clone(), a));
                exprs.push((ctx, b));
            }
        }
    }
    for e in abi.exports.iter() {
        for (axis, g) in e.grid.iter().enumerate() {
            exprs.push((format!("export {} grid[{axis}]", e.symbol), *g));
        }
    }
    for (ctx, expr) in &exprs {
        visit(expr, &mut |node| match node {
            Expr::Param(name) => {
                let resolves = abi.param_index(name).is_some()
                    || abi
                        .exports
                        .iter()
                        .all(|e| e.consts.iter().any(|(c, _)| c == name));
                if !resolves {
                    errs.push(format!(
                        "{ctx}: {name} is neither a param nor a constant of every export"
                    ));
                }
            }
            Expr::DivExact(_, 0) | Expr::DivCeil(_, 0) | Expr::Div(_, 0) => {
                errs.push(format!("{ctx}: division by the constant zero"));
            }
            _ => {}
        });
    }

    // Exports.
    for (i, e) in abi.exports.iter().enumerate() {
        if abi.exports[..i]
            .iter()
            .any(|q| q.arch == e.arch && q.symbol == e.symbol)
        {
            errs.push(format!(
                "export ({}, {}) is declared twice",
                e.arch.name(),
                e.symbol
            ));
        }
        let lds = u64::from(e.dynamic_lds) + u64::from(e.group_segment_fixed);
        if lds > MAX_LDS_BYTES {
            errs.push(format!(
                "export {}: static {} + dynamic {} LDS bytes exceed {MAX_LDS_BYTES}",
                e.symbol, e.group_segment_fixed, e.dynamic_lds
            ));
        }
        if e.block.contains(&0) {
            errs.push(format!(
                "export {}: block {:?} has a zero dimension",
                e.symbol, e.block
            ));
        }
        let threads = block_product(e.block);
        if threads > MAX_BLOCK_THREADS {
            errs.push(format!(
                "export {}: block {:?} has {threads} threads, the limit is {MAX_BLOCK_THREADS}",
                e.symbol, e.block
            ));
        }
    }

    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs.into_iter().map(|r| contract_err(abi, r)).collect())
    }
}

/// A toy contract shared by this crate's unit tests.
#[cfg(test)]
pub(crate) mod toy {
    use crate::{
        Access, Arch, Axis, DomainRuleV1, ExportV1, Expr, FieldKind, FieldV1, FormatId, HiddenKind,
        HiddenV1, KernelAbiV1, NumericContractId, ObjectRefV1, OperandV1, ParamV1, Rule, ScalarTy,
    };

    pub const FMT_X: FormatId = FormatId("toy.x");
    pub const FMT_Y: FormatId = FormatId("toy.y");
    pub const WHY_M: &str = "m must be positive";

    const fn param(name: &'static str) -> ParamV1 {
        ParamV1 {
            name,
            ty: ScalarTy::I32,
            doc: "",
        }
    }

    const fn pointer(name: &'static str, offset: u32, operand: &'static str) -> FieldV1 {
        FieldV1 {
            name,
            offset,
            width: 8,
            kind: FieldKind::Pointer { operand },
        }
    }

    const fn scalar(name: &'static str, offset: u32, value: Expr) -> FieldV1 {
        FieldV1 {
            name,
            offset,
            width: 4,
            kind: FieldKind::Scalar {
                ty: ScalarTy::I32,
                value,
            },
        }
    }

    const fn hidden(offset: u32, width: u32, kind: HiddenKind) -> HiddenV1 {
        HiddenV1 {
            offset,
            width,
            kind,
        }
    }

    /// Two pointer operands, three i32 params (`M`, `S`, `K`), a derived
    /// field `S - 1`, a reserved zero word, the standard hidden block and two
    /// exports with different constants and grids.
    ///
    /// Explicit image (36 bytes): x@0 y@8 m@16 s@20 k@24 s_minus_1@28
    /// reserved@32. Hidden words at 40..96 plus one `Other` word at 96.
    pub static TOY: KernelAbiV1 = KernelAbiV1 {
        id: "toy.abi",
        version: 1,
        wave: 32,
        numeric: NumericContractId("toy.numeric"),
        params: &[param("M"), param("S"), param("K")],
        operands: &[
            OperandV1 {
                name: "x",
                format: FMT_X,
                access: Access::ReadOnly,
                align: 16,
                extent: Some(Expr::Mul(&[
                    Expr::Param("M"),
                    Expr::Param("K"),
                    Expr::Const(2),
                ])),
                nullable: false,
                note: "",
            },
            OperandV1 {
                name: "y",
                format: FMT_Y,
                access: Access::WriteOnly,
                align: 4,
                extent: Some(Expr::Mul(&[
                    Expr::Param("M"),
                    Expr::Param("WIDTH"),
                    Expr::Const(4),
                ])),
                nullable: true,
                note: "",
            },
        ],
        fields: &[
            pointer("x", 0, "x"),
            pointer("y", 8, "y"),
            scalar("m", 16, Expr::Param("M")),
            scalar("s", 20, Expr::Param("S")),
            scalar("k", 24, Expr::Param("K")),
            scalar(
                "s_minus_1",
                28,
                Expr::Sub(&Expr::Param("S"), &Expr::Const(1)),
            ),
            scalar("reserved", 32, Expr::Const(0)),
        ],
        hidden: &[
            hidden(40, 4, HiddenKind::BlockCount(Axis::X)),
            hidden(44, 4, HiddenKind::BlockCount(Axis::Y)),
            hidden(48, 4, HiddenKind::BlockCount(Axis::Z)),
            hidden(52, 2, HiddenKind::GroupSize(Axis::X)),
            hidden(54, 2, HiddenKind::GroupSize(Axis::Y)),
            hidden(56, 2, HiddenKind::GroupSize(Axis::Z)),
            hidden(58, 2, HiddenKind::Remainder(Axis::X)),
            hidden(60, 2, HiddenKind::Remainder(Axis::Y)),
            hidden(62, 2, HiddenKind::Remainder(Axis::Z)),
            hidden(64, 8, HiddenKind::GlobalOffset(Axis::X)),
            hidden(72, 8, HiddenKind::GlobalOffset(Axis::Y)),
            hidden(80, 8, HiddenKind::GlobalOffset(Axis::Z)),
            hidden(88, 2, HiddenKind::GridDims),
            hidden(92, 4, HiddenKind::DynamicLdsSize),
            hidden(96, 8, HiddenKind::Other("hidden_multigrid_sync_arg")),
        ],
        explicit_bytes: 36,
        segment_bytes: 104,
        segment_align: 8,
        domain: &[
            DomainRuleV1 {
                rule: Rule::Positive(Expr::Param("M")),
                why: WHY_M,
            },
            DomainRuleV1 {
                rule: Rule::MultipleOf(Expr::Param("K"), 8),
                why: "k feeds 8-wide loads",
            },
            DomainRuleV1 {
                rule: Rule::AtMost(Expr::Param("S"), 4096),
                why: "s indexes a 4096-entry table",
            },
        ],
        exports: &[
            ExportV1 {
                symbol: "toy_a",
                arch: Arch::Gfx1100,
                object: ObjectRefV1::Embedded {
                    module: "toy_mod",
                    artifact: "kernels/toy.hsaco",
                    sha256: "abc123",
                },
                block: [64, 1, 1],
                dynamic_lds: 0,
                group_segment_fixed: 1024,
                grid: [
                    Expr::DivCeil(&Expr::Param("M"), 4),
                    Expr::Const(1),
                    Expr::Const(1),
                ],
                consts: &[("WIDTH", 1)],
            },
            ExportV1 {
                symbol: "toy_b",
                arch: Arch::Gfx1201,
                object: ObjectRefV1::HipccSource {
                    module: "toy_hip",
                    source: "kernels/src/toy.hip",
                },
                block: [32, 2, 1],
                dynamic_lds: 256,
                group_segment_fixed: 128,
                grid: [Expr::Param("M"), Expr::Param("S"), Expr::Const(1)],
                consts: &[("WIDTH", 2), ("EXTRA", 7)],
            },
        ],
    };
}

#[cfg(test)]
mod tests {
    use super::toy::{FMT_X, FMT_Y, TOY, WHY_M};
    use super::*;
    use crate::{Arch, ObjectRefV1, OperandBinding};

    const GOOD_PARAMS: [u64; 3] = [16, 5, 32];

    fn leak<T>(v: Vec<T>) -> &'static [T] {
        Box::leak(v.into_boxed_slice())
    }

    fn mutate(f: impl FnOnce(&mut KernelAbiV1)) -> KernelAbiV1 {
        let mut a = TOY;
        f(&mut a);
        a
    }

    fn with_fields(f: impl FnOnce(&mut Vec<FieldV1>)) -> KernelAbiV1 {
        mutate(|a| {
            let mut v = a.fields.to_vec();
            f(&mut v);
            a.fields = leak(v);
        })
    }

    fn with_export0(f: impl FnOnce(&mut ExportV1)) -> KernelAbiV1 {
        mutate(|a| {
            let mut v = a.exports.to_vec();
            f(&mut v[0]);
            a.exports = leak(v);
        })
    }

    fn good_ops() -> [OperandBinding; 2] {
        [
            OperandBinding {
                address: 0x1000,
                bytes: Some(1024),
                format: FMT_X,
            },
            OperandBinding {
                address: 0x2004,
                bytes: Some(64),
                format: FMT_Y,
            },
        ]
    }

    fn inv_a<'a>(params: &'a [u64], ops: &'a [OperandBinding]) -> InvocationV1<'a> {
        InvocationV1 {
            arch: Arch::Gfx1100,
            symbol: "toy_a",
            params,
            operands: ops,
            grid: [4, 1, 1],
            block: [64, 1, 1],
            dynamic_lds: 0,
        }
    }

    fn inv_b<'a>(params: &'a [u64], ops: &'a [OperandBinding]) -> InvocationV1<'a> {
        InvocationV1 {
            arch: Arch::Gfx1201,
            symbol: "toy_b",
            params,
            operands: ops,
            grid: [16, 5, 1],
            block: [32, 2, 1],
            dynamic_lds: 256,
        }
    }

    fn reasons(errs: &[AbiError]) -> Vec<String> {
        errs.iter()
            .map(|e| match e {
                AbiError::Contract { reason, .. } => reason.clone(),
                other => panic!("self_check returned non-Contract error {other:?}"),
            })
            .collect()
    }

    fn flags(abi: &KernelAbiV1, needle: &str) -> bool {
        let errs = self_check(abi).expect_err("self_check should fail");
        reasons(&errs).iter().any(|r| r.contains(needle))
    }

    const GOOD_IMAGE: [u8; 36] = [
        0x00, 0x10, 0, 0, 0, 0, 0, 0, // x = 0x1000
        0x04, 0x20, 0, 0, 0, 0, 0, 0, // y = 0x2004
        16, 0, 0, 0, // m
        5, 0, 0, 0, // s
        32, 0, 0, 0, // k
        4, 0, 0, 0, // s - 1
        0, 0, 0, 0, // reserved
    ];

    #[test]
    fn toy_contract_is_self_consistent() {
        assert_eq!(self_check(&TOY), Ok(()));
        assert_eq!(TOY.self_check(), Ok(()));
    }

    #[test]
    fn happy_path_bytes_exact() {
        let ops = good_ops();
        let v = validate_invocation(&TOY, &inv_a(&GOOD_PARAMS, &ops)).unwrap();
        assert_eq!(v.explicit(), &GOOD_IMAGE[..]);
        assert_eq!(v.explicit().len(), TOY.explicit_bytes as usize);
        assert_eq!(v.abi().id, "toy.abi");
        assert_eq!(v.export().symbol, "toy_a");
        assert_eq!(v.params(), &GOOD_PARAMS[..]);
        assert_eq!(v.param("S"), Some(5));
        assert_eq!(v.param("WIDTH"), None);
        assert_eq!(v.grid(), [4, 1, 1]);
        assert_eq!(v.block(), [64, 1, 1]);
        assert_eq!(v.dynamic_lds(), 0);
        assert_eq!(v.unchecked_extents().count(), 0);
    }

    #[test]
    fn second_export_uses_its_own_consts_and_grid() {
        let mut ops = good_ops();
        ops[1].bytes = Some(128); // 16 * WIDTH(2) * 4
        let v = validate_invocation(&TOY, &inv_b(&GOOD_PARAMS, &ops)).unwrap();
        assert_eq!(v.export().symbol, "toy_b");
        assert_eq!(v.explicit(), &GOOD_IMAGE[..]);
        // y needs 128 bytes on toy_b but 64 on toy_a.
        ops[1].bytes = Some(64);
        let err = validate_invocation(&TOY, &inv_b(&GOOD_PARAMS, &ops)).unwrap_err();
        assert_eq!(
            err,
            AbiError::Extent {
                abi: "toy.abi",
                operand: "y",
                need: 128,
                have: 64
            }
        );
    }

    #[test]
    fn unknown_export_by_symbol_and_by_arch() {
        let ops = good_ops();
        let mut i = inv_a(&GOOD_PARAMS, &ops);
        i.symbol = "toy_c";
        assert_eq!(
            validate_invocation(&TOY, &i).unwrap_err(),
            AbiError::UnknownExport {
                abi: "toy.abi",
                arch: "gfx1100",
                symbol: "toy_c".into()
            }
        );
        let mut i = inv_a(&GOOD_PARAMS, &ops);
        i.symbol = "toy_b"; // exists, but for gfx1201
        assert_eq!(
            validate_invocation(&TOY, &i).unwrap_err(),
            AbiError::UnknownExport {
                abi: "toy.abi",
                arch: "gfx1100",
                symbol: "toy_b".into()
            }
        );
    }

    #[test]
    fn param_and_operand_counts() {
        let ops = good_ops();
        assert_eq!(
            validate_invocation(&TOY, &inv_a(&[16, 5], &ops)).unwrap_err(),
            AbiError::ParamCount {
                abi: "toy.abi",
                count: 2,
                expected: 3
            }
        );
        assert_eq!(
            validate_invocation(&TOY, &inv_a(&GOOD_PARAMS, &ops[..1])).unwrap_err(),
            AbiError::OperandCount {
                abi: "toy.abi",
                count: 1,
                expected: 2
            }
        );
    }

    #[test]
    fn param_range() {
        let ops = good_ops();
        let big = 1u64 << 31;
        assert_eq!(
            validate_invocation(&TOY, &inv_a(&[16, big, 32], &ops)).unwrap_err(),
            AbiError::ParamRange {
                abi: "toy.abi",
                name: "S",
                value: big,
                ty: ScalarTy::I32
            }
        );
    }

    #[test]
    fn domain_rule_failure_carries_the_rule_and_reason() {
        let ops = good_ops();
        assert_eq!(
            validate_invocation(&TOY, &inv_a(&[0, 5, 32], &ops)).unwrap_err(),
            AbiError::Domain {
                abi: "toy.abi",
                rule: Rule::Positive(Expr::Param("M")),
                why: WHY_M
            }
        );
        let err = validate_invocation(&TOY, &inv_a(&[16, 5, 12], &ops)).unwrap_err();
        assert!(matches!(
            err,
            AbiError::Domain {
                rule: Rule::MultipleOf(Expr::Param("K"), 8),
                ..
            }
        ));
        let err = validate_invocation(&TOY, &inv_a(&[16, 4097, 32], &ops)).unwrap_err();
        assert!(matches!(
            err,
            AbiError::Domain {
                rule: Rule::AtMost(Expr::Param("S"), 4096),
                ..
            }
        ));
    }

    #[test]
    fn rule_predicates() {
        let e = &TOY.exports[0];
        let p = [16u64, 5, 32];
        let s = Expr::Param("S");
        let h = |r: Rule| rule_holds(&r, &TOY, e, &p).unwrap();
        assert!(h(Rule::Equal(s, Expr::Const(5))));
        assert!(!h(Rule::Equal(s, Expr::Const(6))));
        assert!(h(Rule::OneOf(s, &[1, 5])));
        assert!(!h(Rule::OneOf(s, &[1, 2])));
        assert!(h(Rule::AtLeast(s, 5)));
        assert!(!h(Rule::AtLeast(s, 6)));
        assert!(h(Rule::AtMost(s, 5)));
        assert!(!h(Rule::AtMost(s, 4)));
        assert!(h(Rule::MultipleOf(Expr::Param("K"), 8)));
        assert!(!h(Rule::MultipleOf(s, 2)));
        assert!(h(Rule::Positive(s)));
        let err = rule_holds(&Rule::MultipleOf(s, 0), &TOY, e, &p).unwrap_err();
        assert!(matches!(err, AbiError::Contract { .. }));
        let err = rule_holds(&Rule::Positive(Expr::Param("NOPE")), &TOY, e, &p).unwrap_err();
        assert!(matches!(err, AbiError::Expr { .. }));
    }

    #[test]
    fn launch_block_grid_and_dynamic_lds() {
        let ops = good_ops();
        let mut i = inv_a(&GOOD_PARAMS, &ops);
        i.block = [32, 1, 1];
        assert_eq!(
            validate_invocation(&TOY, &i).unwrap_err(),
            AbiError::Launch {
                abi: "toy.abi",
                what: "block",
                expected: [64, 1, 1],
                got: [32, 1, 1]
            }
        );
        let mut i = inv_a(&GOOD_PARAMS, &ops);
        i.dynamic_lds = 4;
        assert_eq!(
            validate_invocation(&TOY, &i).unwrap_err(),
            AbiError::Launch {
                abi: "toy.abi",
                what: "dynamic_lds",
                expected: [0, 0, 0],
                got: [4, 0, 0]
            }
        );
        let mut i = inv_a(&GOOD_PARAMS, &ops);
        i.grid = [5, 1, 1];
        assert_eq!(
            validate_invocation(&TOY, &i).unwrap_err(),
            AbiError::Launch {
                abi: "toy.abi",
                what: "grid",
                expected: [4, 1, 1],
                got: [5, 1, 1]
            }
        );
    }

    #[test]
    fn operand_format_null_alignment_and_extent() {
        let good = good_ops();

        let mut ops = good;
        ops[0].format = FMT_Y;
        assert_eq!(
            validate_invocation(&TOY, &inv_a(&GOOD_PARAMS, &ops)).unwrap_err(),
            AbiError::Format {
                abi: "toy.abi",
                operand: "x",
                expected: FMT_X,
                got: FMT_Y
            }
        );

        let mut ops = good;
        ops[0].address = 0;
        assert_eq!(
            validate_invocation(&TOY, &inv_a(&GOOD_PARAMS, &ops)).unwrap_err(),
            AbiError::Null {
                abi: "toy.abi",
                operand: "x"
            }
        );

        let mut ops = good;
        ops[0].address = 0x1008;
        assert_eq!(
            validate_invocation(&TOY, &inv_a(&GOOD_PARAMS, &ops)).unwrap_err(),
            AbiError::Misaligned {
                abi: "toy.abi",
                operand: "x",
                address: 0x1008,
                align: 16
            }
        );

        let mut ops = good;
        ops[0].bytes = Some(1023);
        assert_eq!(
            validate_invocation(&TOY, &inv_a(&GOOD_PARAMS, &ops)).unwrap_err(),
            AbiError::Extent {
                abi: "toy.abi",
                operand: "x",
                need: 1024,
                have: 1023
            }
        );
    }

    #[test]
    fn nullable_operand_may_be_null_and_is_not_extent_checked() {
        let mut ops = good_ops();
        ops[1].address = 0;
        ops[1].bytes = Some(0);
        let v = validate_invocation(&TOY, &inv_a(&GOOD_PARAMS, &ops)).unwrap();
        assert_eq!(&v.explicit()[8..16], &[0u8; 8]);
        assert_eq!(v.unchecked_extents().count(), 0);
        // Unbound byte count on a null nullable operand is not flagged either.
        ops[1].bytes = None;
        let v = validate_invocation(&TOY, &inv_a(&GOOD_PARAMS, &ops)).unwrap();
        assert_eq!(v.unchecked_extents().count(), 0);
    }

    #[test]
    fn expression_failure_is_an_expr_error() {
        // S = 0 passes every domain rule but `S - 1` goes negative.
        let ops = good_ops();
        let err = validate_invocation(&TOY, &inv_a(&[16, 0, 32], &ops)).unwrap_err();
        match err {
            AbiError::Expr { abi, expr, reason } => {
                assert_eq!(abi, "toy.abi");
                assert!(expr.contains("Sub"), "{expr}");
                assert!(reason.contains("negative"), "{reason}");
            }
            other => panic!("expected Expr, got {other:?}"),
        }
    }

    #[test]
    fn contract_limits_are_rejected() {
        let ops = good_ops();
        let i = inv_a(&GOOD_PARAMS, &ops);
        let too_big = mutate(|a| a.explicit_bytes = MAX_EXPLICIT_BYTES as u32 + 1);
        assert!(matches!(
            validate_invocation(&too_big, &i).unwrap_err(),
            AbiError::Contract { .. }
        ));
        let many_fields = mutate(|a| a.fields = leak(vec![a.fields[2]; MAX_FIELDS + 1]));
        assert!(matches!(
            validate_invocation(&many_fields, &i).unwrap_err(),
            AbiError::Contract { .. }
        ));
        let many_params = mutate(|a| a.params = leak(vec![a.params[0]; MAX_FIELDS + 1]));
        assert!(matches!(
            validate_invocation(&many_params, &i).unwrap_err(),
            AbiError::Contract { .. }
        ));
        let many_operands = mutate(|a| a.operands = leak(vec![a.operands[0]; 65]));
        assert!(matches!(
            validate_invocation(&many_operands, &i).unwrap_err(),
            AbiError::Contract { .. }
        ));
        let zero_align = mutate(|a| {
            let mut ops = a.operands.to_vec();
            ops[0].align = 0;
            a.operands = leak(ops);
        });
        assert!(matches!(
            validate_invocation(&zero_align, &i).unwrap_err(),
            AbiError::Contract { .. }
        ));
    }

    #[test]
    fn malformed_fields_are_contract_errors_at_encode() {
        let ops = good_ops();
        let i = inv_a(&GOOD_PARAMS, &ops);
        let narrow_ptr = with_fields(|f| f[0].width = 4);
        assert!(matches!(
            validate_invocation(&narrow_ptr, &i).unwrap_err(),
            AbiError::Contract { .. }
        ));
        let wide_scalar = with_fields(|f| f[2].width = 8);
        assert!(matches!(
            validate_invocation(&wide_scalar, &i).unwrap_err(),
            AbiError::Contract { .. }
        ));
        let outside = with_fields(|f| f[2].offset = 34);
        assert!(matches!(
            validate_invocation(&outside, &i).unwrap_err(),
            AbiError::Contract { .. }
        ));
        let unknown_op = with_fields(|f| f[0].kind = FieldKind::Pointer { operand: "nope" });
        assert!(matches!(
            validate_invocation(&unknown_op, &i).unwrap_err(),
            AbiError::Contract { .. }
        ));
    }

    #[test]
    fn scalar_types_encode_and_range_check() {
        let ops = good_ops();
        let i = inv_a(&GOOD_PARAMS, &ops);

        // I32 field: 2^31 is rejected, named after the field.
        let bad_i32 = with_fields(|f| {
            f[5].kind = FieldKind::Scalar {
                ty: ScalarTy::I32,
                value: Expr::Const(1 << 31),
            }
        });
        assert_eq!(
            validate_invocation(&bad_i32, &i).unwrap_err(),
            AbiError::ParamRange {
                abi: "toy.abi",
                name: "s_minus_1",
                value: 1 << 31,
                ty: ScalarTy::I32
            }
        );

        // U32 field accepts u32::MAX, rejects 2^32.
        let u32_max = with_fields(|f| {
            f[6].kind = FieldKind::Scalar {
                ty: ScalarTy::U32,
                value: Expr::Const(u64::from(u32::MAX)),
            }
        });
        let v = validate_invocation(&u32_max, &i).unwrap();
        assert_eq!(&v.explicit()[32..36], &[0xff; 4]);
        let u32_over = with_fields(|f| {
            f[6].kind = FieldKind::Scalar {
                ty: ScalarTy::U32,
                value: Expr::Const(1 << 32),
            }
        });
        assert!(matches!(
            validate_invocation(&u32_over, &i).unwrap_err(),
            AbiError::ParamRange {
                ty: ScalarTy::U32,
                ..
            }
        ));

        // U64 field is 8 bytes wide.
        let u64_field = mutate(|a| {
            let mut f = a.fields.to_vec();
            f[6].width = 8;
            f[6].kind = FieldKind::Scalar {
                ty: ScalarTy::U64,
                value: Expr::Const(u64::MAX),
            };
            a.fields = leak(f);
            a.explicit_bytes = 40;
        });
        let v = validate_invocation(&u64_field, &i).unwrap();
        assert_eq!(v.explicit().len(), 40);
        assert_eq!(&v.explicit()[32..40], &[0xff; 8]);
        assert_eq!(&v.explicit()[..32], &GOOD_IMAGE[..32]);
    }

    #[test]
    fn unchecked_extents_reports_the_unbound_operand() {
        let mut ops = good_ops();
        ops[1].bytes = None;
        let v = validate_invocation(&TOY, &inv_a(&GOOD_PARAMS, &ops)).unwrap();
        let names: Vec<&str> = v.unchecked_extents().map(|o| o.name).collect();
        assert_eq!(names, ["y"]);

        ops[0].bytes = None;
        let v = validate_invocation(&TOY, &inv_a(&GOOD_PARAMS, &ops)).unwrap();
        let names: Vec<&str> = v.unchecked_extents().map(|o| o.name).collect();
        assert_eq!(names, ["x", "y"]);
    }

    #[test]
    fn kernel_params_point_into_the_explicit_image_in_offset_order() {
        let ops = good_ops();
        let mut v = validate_invocation(&TOY, &inv_a(&GOOD_PARAMS, &ops)).unwrap();
        let mut out = [core::ptr::null_mut::<c_void>(); 8];
        let n = v.kernel_params(&mut out).unwrap();
        assert_eq!(n, TOY.fields.len());
        let base = v.explicit().as_ptr() as usize;
        for (p, f) in out.iter().zip(TOY.fields) {
            assert_eq!(*p as usize, base + f.offset as usize, "{}", f.name);
        }
        assert!(out[n].is_null(), "entries past n are untouched");
        assert!(out[..n]
            .windows(2)
            .all(|w| (w[0] as usize) < (w[1] as usize)));

        let mut short = [core::ptr::null_mut::<c_void>(); 6];
        let err = v.kernel_params(&mut short).unwrap_err();
        let word = core::mem::size_of::<*mut c_void>();
        assert_eq!(
            err,
            AbiError::Buffer {
                abi: "toy.abi",
                need: 7 * word,
                got: 6 * word
            }
        );
    }

    fn hidden_map(v: &ValidatedInvocationV1<'_>) -> Vec<(HiddenKind, u64)> {
        v.hidden_expected().map(|(h, val)| (h.kind, val)).collect()
    }

    fn grid_dims(v: &ValidatedInvocationV1<'_>) -> u64 {
        hidden_map(v)
            .into_iter()
            .find(|(k, _)| *k == HiddenKind::GridDims)
            .map(|(_, v)| v)
            .unwrap()
    }

    #[test]
    fn hidden_expected_values() {
        use crate::Axis::{X, Y, Z};
        let mut ops = good_ops();
        ops[1].bytes = Some(128);
        let v = validate_invocation(&TOY, &inv_b(&GOOD_PARAMS, &ops)).unwrap();
        let got = hidden_map(&v);
        let want = vec![
            (HiddenKind::BlockCount(X), 16),
            (HiddenKind::BlockCount(Y), 5),
            (HiddenKind::BlockCount(Z), 1),
            (HiddenKind::GroupSize(X), 32),
            (HiddenKind::GroupSize(Y), 2),
            (HiddenKind::GroupSize(Z), 1),
            (HiddenKind::Remainder(X), 0),
            (HiddenKind::Remainder(Y), 0),
            (HiddenKind::Remainder(Z), 0),
            (HiddenKind::GlobalOffset(X), 0),
            (HiddenKind::GlobalOffset(Y), 0),
            (HiddenKind::GlobalOffset(Z), 0),
            (HiddenKind::GridDims, 2),
            (HiddenKind::DynamicLdsSize, 256),
        ];
        assert_eq!(got, want, "Other kinds are skipped");
        // The returned HiddenV1 is the contract's word.
        let first = v.hidden_expected().next().unwrap().0;
        assert_eq!(first, TOY.hidden[0]);
    }

    #[test]
    fn grid_dims_rule() {
        let ops = good_ops();
        // 1: 1-D grid and block.
        let v = validate_invocation(&TOY, &inv_a(&GOOD_PARAMS, &ops)).unwrap();
        assert_eq!(grid_dims(&v), 1);
        // 2: y extent in the block (toy_b).
        let mut ops_b = good_ops();
        ops_b[1].bytes = Some(128);
        let v = validate_invocation(&TOY, &inv_b(&GOOD_PARAMS, &ops_b)).unwrap();
        assert_eq!(grid_dims(&v), 2);
        // 3: z extent in the grid.
        let grid_z = with_export0(|e| e.grid[2] = Expr::Const(3));
        let mut i = inv_a(&GOOD_PARAMS, &ops);
        i.grid = [4, 1, 3];
        let v = validate_invocation(&grid_z, &i).unwrap();
        assert_eq!(grid_dims(&v), 3);
        // 3: z extent in the block only.
        let block_z = with_export0(|e| e.block = [8, 1, 2]);
        let mut i = inv_a(&GOOD_PARAMS, &ops);
        i.block = [8, 1, 2];
        let v = validate_invocation(&block_z, &i).unwrap();
        assert_eq!(grid_dims(&v), 3);
        // 2: y extent in the grid only.
        let grid_y = with_export0(|e| e.grid[1] = Expr::Const(2));
        let mut i = inv_a(&GOOD_PARAMS, &ops);
        i.grid = [4, 2, 1];
        let v = validate_invocation(&grid_y, &i).unwrap();
        assert_eq!(grid_dims(&v), 2);
    }

    #[test]
    fn encode_into_copies_then_zero_fills_and_rejects_short_buffers() {
        let ops = good_ops();
        let v = validate_invocation(&TOY, &inv_a(&GOOD_PARAMS, &ops)).unwrap();

        let mut exact = [0xAAu8; 36];
        encode_into(&v, &mut exact).unwrap();
        assert_eq!(exact, GOOD_IMAGE);

        let mut padded = [0xAAu8; 64];
        encode_into(&v, &mut padded).unwrap();
        assert_eq!(&padded[..36], &GOOD_IMAGE[..]);
        assert!(padded[36..].iter().all(|&b| b == 0));

        let mut short = [0xAAu8; 35];
        assert_eq!(
            encode_into(&v, &mut short).unwrap_err(),
            AbiError::Buffer {
                abi: "toy.abi",
                need: 36,
                got: 35
            }
        );
        assert!(
            short.iter().all(|&b| b == 0xAA),
            "a rejected call writes nothing"
        );
    }

    #[test]
    fn decode_params_round_trips_the_encoded_image() {
        let ops = good_ops();
        let v = validate_invocation(&TOY, &inv_a(&GOOD_PARAMS, &ops)).unwrap();
        let d = decode_params(&TOY, v.explicit()).unwrap();
        assert_eq!(
            d,
            DecodedV1 {
                params: vec![Some(16), Some(5), Some(32)],
                addresses: vec![0x1000, 0x2004],
            }
        );
        // A longer buffer (kernarg segment) decodes the same.
        let mut long = vec![0u8; 104];
        encode_into(&v, &mut long).unwrap();
        assert_eq!(decode_params(&TOY, &long).unwrap(), d);
    }

    #[test]
    fn decode_params_rejects_tampered_derived_and_reserved_words() {
        let mut image = GOOD_IMAGE;
        image[28] = 9; // s_minus_1 should be 4
        let err = decode_params(&TOY, &image).unwrap_err();
        match err {
            AbiError::Expr { reason, .. } => assert!(reason.contains("s_minus_1"), "{reason}"),
            other => panic!("expected Expr, got {other:?}"),
        }
        let mut image = GOOD_IMAGE;
        image[32] = 1; // reserved word must be zero
        assert!(matches!(
            decode_params(&TOY, &image).unwrap_err(),
            AbiError::Expr { .. }
        ));
    }

    #[test]
    fn decode_params_rejects_short_input_and_out_of_range_scalars() {
        assert_eq!(
            decode_params(&TOY, &GOOD_IMAGE[..35]).unwrap_err(),
            AbiError::Buffer {
                abi: "toy.abi",
                need: 36,
                got: 35
            }
        );
        let mut image = GOOD_IMAGE;
        image[19] = 0x80; // m = 0x8000_0010 > i32::MAX
        assert!(matches!(
            decode_params(&TOY, &image).unwrap_err(),
            AbiError::ParamRange { name: "m", .. }
        ));
    }

    #[test]
    fn decode_params_rejects_conflicting_reads_of_one_param() {
        let dup = with_fields(|f| {
            f[4].kind = FieldKind::Scalar {
                ty: ScalarTy::I32,
                value: Expr::Param("S"),
            }
        });
        let err = decode_params(&dup, &GOOD_IMAGE).unwrap_err(); // k field holds 32, s holds 5
        match err {
            AbiError::Expr { reason, .. } => assert!(reason.contains("conflicting"), "{reason}"),
            other => panic!("expected Expr, got {other:?}"),
        }
    }

    #[test]
    fn decode_params_skips_fields_that_name_export_constants_or_unknown_params() {
        // `reserved` now reads the export constant WIDTH; the image holds 0 and
        // the first export's WIDTH is 1, yet decoding must not compare them.
        let konst = with_fields(|f| {
            f[6].kind = FieldKind::Scalar {
                ty: ScalarTy::I32,
                value: Expr::Param("WIDTH"),
            }
        });
        let d = decode_params(&konst, &GOOD_IMAGE).unwrap();
        assert_eq!(d.params, vec![Some(16), Some(5), Some(32)]);

        // `s_minus_1` depends on S, which no field carries directly any more.
        let no_s = with_fields(|f| {
            f[3].kind = FieldKind::Scalar {
                ty: ScalarTy::I32,
                value: Expr::Const(5),
            }
        });
        let mut image = GOOD_IMAGE;
        image[28] = 0xEE; // not checkable without S, so not rejected
        let d = decode_params(&no_s, &image).unwrap();
        assert_eq!(d.params, vec![Some(16), None, Some(32)]);
    }

    #[test]
    fn self_check_flags_overlap() {
        let abi = with_fields(|f| f[4].offset = 20); // k on top of s
        assert!(flags(&abi, "overlaps"));
    }

    #[test]
    fn self_check_flags_misalignment() {
        let abi = with_fields(|f| f[6].offset = 34);
        assert!(flags(&abi, "naturally aligned"));
        let abi = mutate(|a| {
            let mut h = a.hidden.to_vec();
            h[3].offset = 53; // 2-byte word at an odd offset
            a.hidden = leak(h);
        });
        assert!(flags(&abi, "naturally aligned"));
    }

    #[test]
    fn self_check_flags_words_outside_the_segment() {
        let abi = mutate(|a| a.segment_bytes = 90);
        assert!(flags(&abi, "outside"));
    }

    #[test]
    fn self_check_flags_unknown_operand() {
        let abi = with_fields(|f| f[0].kind = FieldKind::Pointer { operand: "nope" });
        assert!(flags(&abi, "unknown operand nope"));
    }

    #[test]
    fn self_check_flags_operand_carried_twice_or_never() {
        let abi = with_fields(|f| f[1].kind = FieldKind::Pointer { operand: "x" });
        assert!(flags(&abi, "operand x is carried by 2 pointer fields"));
        assert!(flags(&abi, "operand y is not carried"));
    }

    #[test]
    fn self_check_flags_unknown_param_names() {
        let abi = with_fields(|f| {
            f[2].kind = FieldKind::Scalar {
                ty: ScalarTy::I32,
                value: Expr::Param("NOPE"),
            }
        });
        assert!(flags(&abi, "NOPE"));
        // EXTRA exists in toy_b only: not a constant of every export.
        let abi = with_fields(|f| {
            f[2].kind = FieldKind::Scalar {
                ty: ScalarTy::I32,
                value: Expr::Param("EXTRA"),
            }
        });
        assert!(flags(&abi, "EXTRA"));
    }

    #[test]
    fn self_check_flags_explicit_bytes_mismatch() {
        let abi = mutate(|a| a.explicit_bytes = 40);
        assert!(flags(&abi, "explicit_bytes is 40"));
    }

    #[test]
    fn self_check_flags_lds_over_the_limit() {
        let abi = with_export0(|e| {
            e.dynamic_lds = 65536 - 1024 + 1;
            e.group_segment_fixed = 1024;
        });
        assert!(flags(&abi, "LDS"));
        // Exactly at the limit is fine.
        let abi = with_export0(|e| {
            e.dynamic_lds = 65536 - 1024;
            e.group_segment_fixed = 1024;
        });
        assert_eq!(self_check(&abi), Ok(()));
    }

    #[test]
    fn self_check_flags_duplicate_exports() {
        let abi = mutate(|a| a.exports = leak(vec![a.exports[0], a.exports[0]]));
        assert!(flags(&abi, "declared twice"));
    }

    #[test]
    fn self_check_flags_block_and_alignment_problems() {
        let abi = with_export0(|e| e.block = [1024, 2, 1]);
        assert!(flags(&abi, "threads"));
        let abi = with_export0(|e| e.block = [0, 1, 1]);
        assert!(flags(&abi, "zero dimension"));
        let abi = mutate(|a| a.segment_align = 12);
        assert!(flags(&abi, "segment_align"));
        let abi = mutate(|a| {
            let mut o = a.operands.to_vec();
            o[0].align = 24;
            a.operands = leak(o);
        });
        assert!(flags(&abi, "alignment 24"));
    }

    #[test]
    fn self_check_flags_unsorted_hidden_scalar_widths_and_zero_divisors() {
        let abi = mutate(|a| {
            let mut h = a.hidden.to_vec();
            h.swap(0, 1);
            a.hidden = leak(h);
        });
        assert!(flags(&abi, "hidden words are not sorted"));
        let abi = with_fields(|f| f[2].width = 8);
        assert!(flags(&abi, "needs 4"));
        let abi = with_fields(|f| {
            f[2].kind = FieldKind::Scalar {
                ty: ScalarTy::I32,
                value: Expr::Div(&Expr::Param("M"), 0),
            }
        });
        assert!(flags(&abi, "division by the constant zero"));
    }

    #[test]
    fn self_check_collects_every_error() {
        let abi = mutate(|a| {
            a.explicit_bytes = 40;
            a.segment_align = 3;
            a.fields = leak({
                let mut f = a.fields.to_vec();
                f[0].kind = FieldKind::Pointer { operand: "nope" };
                f
            });
        });
        let errs = self_check(&abi).unwrap_err();
        assert!(errs.len() >= 4, "{errs:?}");
        assert!(errs
            .iter()
            .all(|e| matches!(e, AbiError::Contract { abi: "toy.abi", .. })));
    }

    #[test]
    fn digest_is_stable_and_sensitive() {
        let a = TOY.digest();
        assert_eq!(a, TOY.digest());
        assert_eq!(a.len(), 64);
        assert!(a
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        let bumped = mutate(|t| t.version = 2);
        assert_ne!(a, bumped.digest());
        let moved = with_fields(|f| f[4].offset = 25);
        assert_ne!(a, moved.digest());
        let other_object = with_export0(|e| {
            e.object = ObjectRefV1::Embedded {
                module: "toy_mod",
                artifact: "kernels/toy.hsaco",
                sha256: "abc124",
            }
        });
        assert_ne!(a, other_object.digest());
    }
}
