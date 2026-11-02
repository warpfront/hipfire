// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! C2: HSA metadata MessagePack codec.
//!
//! The `NT_AMDGPU_METADATA` note is one msgpack map whose `amdhsa.kernels` array holds
//! one map per kernel. Each kernel map is kept as its verbatim byte slice
//! (`HsaKernelMetadata::raw_msgpack`) plus the typed `KernelMeta` view. Serialisation
//! re-encodes the slice's value tree with every typed field written back, so keys the
//! typed view does not model (`.language`, `.args[].address_space`, …) survive and an
//! unedited kernel re-encodes to its original bytes. Parsing rejects any slice whose
//! re-encoding differs, so that byte identity holds for every accepted input.

use peacemaker_ir::descriptor::KernelDescriptor;
use peacemaker_ir::envelope::{DocEntry, MetadataDoc};
use peacemaker_ir::metadata::{HsaKernelMetadata, Kernarg, KernelMeta};
use rmpv::Value;

/// Top-level key of the per-kernel array in the metadata document.
pub const KERNELS_KEY: &str = "amdhsa.kernels";

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum MetadataError {
    #[error("msgpack decode failed at byte {offset}: {reason}")]
    Decode { offset: usize, reason: String },
    #[error("{0} trailing bytes after the msgpack value")]
    Trailing(usize),
    #[error("{what} is not a msgpack map")]
    NotMap { what: String },
    #[error("required key {key} is missing")]
    Missing { key: String },
    #[error("key {key} has the wrong type (expected {expected})")]
    WrongType { key: String, expected: &'static str },
    #[error("key {key} value {value} is out of range")]
    OutOfRange { key: String, value: String },
    #[error("duplicate key {key}")]
    Duplicate { key: String },
    #[error(
        "{what} is not canonically encoded msgpack; re-serialisation would not reproduce its bytes"
    )]
    NonCanonical { what: String },
    #[error("metadata document has no {KERNELS_KEY} array")]
    NoKernels,
    #[error("metadata document holds {expected} kernel maps, {got} were supplied")]
    KernelCount { expected: usize, got: usize },
    #[error("kernel {kernel}: {reason}")]
    Inconsistent { kernel: String, reason: String },
}

/// Codec for the metadata note payload (`MetadataDoc::parse(..)` with this trait in scope).
/// The skeleton keeps every non-kernel top-level key and value as its own msgpack bytes;
/// the kernel maps are holes filled from typed `HsaKernelMetadata` on write.
pub trait MetadataDocCodec: Sized {
    /// Splits a note payload into the document skeleton and the kernels' metadata in array
    /// order. Fails unless `write` of the result reproduces `desc` exactly.
    fn parse(desc: &[u8]) -> Result<(Self, Vec<HsaKernelMetadata>), MetadataError>;
    /// Number of kernel maps the document holds.
    fn kernel_count(&self) -> usize;
    /// Re-encodes the document, serialising each kernel (array order) from its typed metadata.
    fn write(&self, kernels: &[&HsaKernelMetadata]) -> Result<Vec<u8>, MetadataError>;
}

impl MetadataDocCodec for MetadataDoc {
    fn parse(desc: &[u8]) -> Result<(MetadataDoc, Vec<HsaKernelMetadata>), MetadataError> {
        let mut pos = 0;
        let count = read_container_len(desc, &mut pos, Container::Map, "metadata document")?;
        let mut entries = Vec::with_capacity(count);
        let mut kernels = None;
        for _ in 0..count {
            let key_start = pos;
            let key = read_value_at(desc, &mut pos)?;
            if key.as_str() == Some(KERNELS_KEY) {
                if kernels.is_some() {
                    return Err(MetadataError::Duplicate {
                        key: KERNELS_KEY.into(),
                    });
                }
                let n = read_container_len(desc, &mut pos, Container::Array, KERNELS_KEY)?;
                let mut list = Vec::with_capacity(n);
                for _ in 0..n {
                    let start = pos;
                    read_value_at(desc, &mut pos)?;
                    list.push(parse_kernel(&desc[start..pos])?);
                }
                entries.push(DocEntry::Kernels(n));
                kernels = Some(list);
            } else {
                let key_bytes = &desc[key_start..pos];
                if entries
                    .iter()
                    .any(|e| matches!(e, DocEntry::Raw { key, .. } if key == key_bytes))
                {
                    return Err(MetadataError::Duplicate {
                        key: key.to_string(),
                    });
                }
                let value_start = pos;
                read_value_at(desc, &mut pos)?;
                entries.push(DocEntry::Raw {
                    key: key_bytes.to_vec(),
                    value: desc[value_start..pos].to_vec(),
                });
            }
        }
        if pos != desc.len() {
            return Err(MetadataError::Trailing(desc.len() - pos));
        }
        let kernels = kernels.ok_or(MetadataError::NoKernels)?;
        let doc = MetadataDoc { entries };
        if doc.write(&kernels.iter().collect::<Vec<_>>())? != desc {
            return Err(MetadataError::NonCanonical {
                what: "metadata document".into(),
            });
        }
        Ok((doc, kernels))
    }

    fn kernel_count(&self) -> usize {
        self.entries
            .iter()
            .find_map(|e| match e {
                DocEntry::Kernels(n) => Some(*n),
                DocEntry::Raw { .. } => None,
            })
            .unwrap_or(0)
    }

    fn write(&self, kernels: &[&HsaKernelMetadata]) -> Result<Vec<u8>, MetadataError> {
        let expected = self.kernel_count();
        if kernels.len() != expected {
            return Err(MetadataError::KernelCount {
                expected,
                got: kernels.len(),
            });
        }
        let mut out = Vec::new();
        write_container_len(&mut out, self.entries.len(), Container::Map);
        for entry in &self.entries {
            match entry {
                DocEntry::Kernels(n) => {
                    encode_into(&mut out, &Value::from(KERNELS_KEY));
                    write_container_len(&mut out, *n, Container::Array);
                    for kernel in kernels {
                        out.extend(serialize_kernel(kernel)?);
                    }
                }
                DocEntry::Raw { key, value } => {
                    out.extend_from_slice(key);
                    out.extend_from_slice(value);
                }
            }
        }
        Ok(out)
    }
}

/// Parses one kernel's metadata map. The returned `raw_msgpack` is `raw` verbatim.
pub fn parse_kernel(raw: &[u8]) -> Result<HsaKernelMetadata, MetadataError> {
    let value = decode_exact(raw)?;
    let parsed = kernel_meta(&value)?;
    let meta = HsaKernelMetadata {
        raw_msgpack: raw.to_vec(),
        parsed,
    };
    if serialize_kernel(&meta)? != raw {
        return Err(MetadataError::NonCanonical {
            what: format!("metadata map of kernel {}", meta.parsed.name),
        });
    }
    Ok(meta)
}

/// Serialises a kernel map: the value tree of `raw_msgpack` (an empty map when there is no
/// raw slice) with every `KernelMeta` field written into it. Existing keys keep their
/// position and flag representation (bool vs 0/1 integer); absent keys are inserted in
/// sorted position only when required or when the typed value differs from the default.
pub fn serialize_kernel(meta: &HsaKernelMetadata) -> Result<Vec<u8>, MetadataError> {
    let mut value = if meta.raw_msgpack.is_empty() {
        Value::Map(Vec::new())
    } else {
        decode_exact(&meta.raw_msgpack)?
    };
    apply(&mut value, &meta.parsed)?;
    let mut out = Vec::with_capacity(meta.raw_msgpack.len());
    encode_into(&mut out, &value);
    Ok(out)
}

/// Cross-checks the HSA symbol, wave, private-segment size, and VGPR allocation.
/// On gfx1201 the descriptor may reserve more VGPRs than `.vgpr_count`
/// (AMDGPUUsage only defines the latter as an exact register count through GFX9).
pub fn check(meta: &KernelMeta, kernel: &str, kd: &KernelDescriptor) -> Result<(), MetadataError> {
    let fail = |reason: String| {
        Err(MetadataError::Inconsistent {
            kernel: kernel.into(),
            reason,
        })
    };
    if meta.symbol != format!("{kernel}.kd") {
        return fail(format!(".symbol {} is not {kernel}.kd", meta.symbol));
    }
    let wave32 = kd.kernel_code_properties.wave32();
    let wave = if wave32 { 32 } else { 64 };
    if meta.wavefront_size != wave {
        return fail(format!(
            ".wavefront_size {} but the descriptor selects wave{wave}",
            meta.wavefront_size
        ));
    }
    if meta.private_segment_fixed_size != kd.private_segment_fixed_size {
        return fail(format!(
            ".private_segment_fixed_size {} but the descriptor allocates {}",
            meta.private_segment_fixed_size, kd.private_segment_fixed_size
        ));
    }
    let allocated = kd.compute_pgm_rsrc1.next_free_vgpr(wave32);
    if meta.vgpr_count > allocated {
        return fail(format!(
            ".vgpr_count {} exceeds the descriptor allocation {allocated}",
            meta.vgpr_count
        ));
    }
    Ok(())
}

fn kernel_meta(value: &Value) -> Result<KernelMeta, MetadataError> {
    let map = as_map(value, "kernel metadata")?;
    let args = match get(map, ".args") {
        None => Vec::new(),
        Some(Value::Array(items)) => items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let arg = as_map(item, &format!(".args[{i}]"))?;
                let key = |k: &str| format!(".args[{i}]{k}");
                Ok(Kernarg {
                    name: opt_str(arg, ".name", &key(".name"))?.unwrap_or_default(),
                    size: req_u32(arg, ".size", &key(".size"))?,
                    offset: req_u32(arg, ".offset", &key(".offset"))?,
                    value_kind: req_str(arg, ".value_kind", &key(".value_kind"))?,
                    address_space: opt_str(arg, ".address_space", &key(".address_space"))?,
                })
            })
            .collect::<Result<_, MetadataError>>()?,
        Some(_) => {
            return Err(MetadataError::WrongType {
                key: ".args".into(),
                expected: "array",
            })
        }
    };
    Ok(KernelMeta {
        name: req_str(map, ".name", ".name")?,
        symbol: req_str(map, ".symbol", ".symbol")?,
        args,
        kernarg_segment_size: req_u32(map, ".kernarg_segment_size", ".kernarg_segment_size")?,
        kernarg_segment_align: req_u32(map, ".kernarg_segment_align", ".kernarg_segment_align")?,
        group_segment_fixed_size: req_u32(
            map,
            ".group_segment_fixed_size",
            ".group_segment_fixed_size",
        )?,
        private_segment_fixed_size: req_u32(
            map,
            ".private_segment_fixed_size",
            ".private_segment_fixed_size",
        )?,
        vgpr_count: req_u32(map, ".vgpr_count", ".vgpr_count")?,
        sgpr_count: req_u32(map, ".sgpr_count", ".sgpr_count")?,
        wavefront_size: req_u32(map, ".wavefront_size", ".wavefront_size")?,
        max_flat_workgroup_size: req_u32(
            map,
            ".max_flat_workgroup_size",
            ".max_flat_workgroup_size",
        )?,
        workgroup_processor_mode: opt_flag(map, ".workgroup_processor_mode")?,
        uniform_work_group_size: opt_flag(map, ".uniform_work_group_size")?,
        uses_dynamic_stack: opt_flag(map, ".uses_dynamic_stack")?,
        sgpr_spill_count: opt_u32(map, ".sgpr_spill_count")?.unwrap_or(0),
        vgpr_spill_count: opt_u32(map, ".vgpr_spill_count")?.unwrap_or(0),
    })
}

/// How an absent flag key is spelled when a non-default value forces its insertion
/// (matching LLVM's emitter: `.uses_dynamic_stack` is a bool, the others are integers).
#[derive(Clone, Copy)]
enum FlagRepr {
    Bool,
    Int,
}

fn apply(value: &mut Value, meta: &KernelMeta) -> Result<(), MetadataError> {
    let Value::Map(map) = value else {
        return Err(MetadataError::NotMap {
            what: "kernel metadata".into(),
        });
    };
    put(map, ".name", Value::from(meta.name.as_str()), true);
    put(map, ".symbol", Value::from(meta.symbol.as_str()), true);
    for (key, v) in [
        (".kernarg_segment_size", meta.kernarg_segment_size),
        (".kernarg_segment_align", meta.kernarg_segment_align),
        (".group_segment_fixed_size", meta.group_segment_fixed_size),
        (
            ".private_segment_fixed_size",
            meta.private_segment_fixed_size,
        ),
        (".vgpr_count", meta.vgpr_count),
        (".sgpr_count", meta.sgpr_count),
        (".wavefront_size", meta.wavefront_size),
        (".max_flat_workgroup_size", meta.max_flat_workgroup_size),
    ] {
        put(map, key, Value::from(v), true);
    }
    for (key, v) in [
        (".sgpr_spill_count", meta.sgpr_spill_count),
        (".vgpr_spill_count", meta.vgpr_spill_count),
    ] {
        put(map, key, Value::from(v), v != 0);
    }
    for (key, v, repr) in [
        (
            ".workgroup_processor_mode",
            meta.workgroup_processor_mode,
            FlagRepr::Int,
        ),
        (
            ".uniform_work_group_size",
            meta.uniform_work_group_size,
            FlagRepr::Int,
        ),
        (
            ".uses_dynamic_stack",
            meta.uses_dynamic_stack,
            FlagRepr::Bool,
        ),
    ] {
        put_flag(map, key, v, repr)?;
    }
    if meta.args.is_empty() && get(map, ".args").is_none() {
        return Ok(());
    }
    if get(map, ".args").is_none() {
        put(map, ".args", Value::Array(Vec::new()), true);
    }
    let Some(Value::Array(items)) = get_mut(map, ".args") else {
        return Err(MetadataError::WrongType {
            key: ".args".into(),
            expected: "array",
        });
    };
    items.truncate(meta.args.len());
    for (i, arg) in meta.args.iter().enumerate() {
        if i == items.len() {
            items.push(Value::Map(Vec::new()));
        }
        let Value::Map(fields) = &mut items[i] else {
            return Err(MetadataError::NotMap {
                what: format!(".args[{i}]"),
            });
        };
        let named = !arg.name.is_empty();
        match &arg.address_space {
            Some(space) => put(fields, ".address_space", Value::from(space.as_str()), true),
            None => fields.retain(|(k, _)| k.as_str() != Some(".address_space")),
        }
        put(fields, ".name", Value::from(arg.name.as_str()), named);
        put(fields, ".offset", Value::from(arg.offset), true);
        put(fields, ".size", Value::from(arg.size), true);
        put(
            fields,
            ".value_kind",
            Value::from(arg.value_kind.as_str()),
            true,
        );
    }
    Ok(())
}

/// Replaces `key` in place, or inserts it in sorted key order when `insert` is set
/// (LLVM's msgpack document keeps map keys sorted).
fn put(map: &mut Vec<(Value, Value)>, key: &str, value: Value, insert: bool) {
    if let Some(slot) = get_mut(map, key) {
        *slot = value;
    } else if insert {
        let at = map
            .iter()
            .position(|(k, _)| k.as_str().is_some_and(|k| k > key))
            .unwrap_or(map.len());
        map.insert(at, (Value::from(key), value));
    }
}

fn put_flag(
    map: &mut Vec<(Value, Value)>,
    key: &str,
    flag: bool,
    repr: FlagRepr,
) -> Result<(), MetadataError> {
    let value = match get(map, key) {
        Some(Value::Boolean(_)) => Value::Boolean(flag),
        Some(Value::Integer(_)) => Value::from(u8::from(flag)),
        Some(_) => {
            return Err(MetadataError::WrongType {
                key: key.into(),
                expected: "bool or 0/1 integer",
            })
        }
        None if !flag => return Ok(()),
        None => match repr {
            FlagRepr::Bool => Value::Boolean(true),
            FlagRepr::Int => Value::from(1u8),
        },
    };
    put(map, key, value, true);
    Ok(())
}

fn as_map<'a>(value: &'a Value, what: &str) -> Result<&'a [(Value, Value)], MetadataError> {
    let Value::Map(map) = value else {
        return Err(MetadataError::NotMap { what: what.into() });
    };
    for (i, (key, _)) in map.iter().enumerate() {
        if map[..i].iter().any(|(k, _)| k == key) {
            return Err(MetadataError::Duplicate {
                key: format!("{what}: {key}"),
            });
        }
    }
    Ok(map)
}

fn get<'a>(map: &'a [(Value, Value)], key: &str) -> Option<&'a Value> {
    map.iter()
        .find(|(k, _)| k.as_str() == Some(key))
        .map(|(_, v)| v)
}

fn get_mut<'a>(map: &'a mut [(Value, Value)], key: &str) -> Option<&'a mut Value> {
    map.iter_mut()
        .find(|(k, _)| k.as_str() == Some(key))
        .map(|(_, v)| v)
}

fn opt_str(map: &[(Value, Value)], key: &str, path: &str) -> Result<Option<String>, MetadataError> {
    match get(map, key) {
        None => Ok(None),
        Some(v) => v
            .as_str()
            .map(|s| Some(s.to_owned()))
            .ok_or_else(|| MetadataError::WrongType {
                key: path.into(),
                expected: "UTF-8 string",
            }),
    }
}

fn req_str(map: &[(Value, Value)], key: &str, path: &str) -> Result<String, MetadataError> {
    opt_str(map, key, path)?.ok_or_else(|| MetadataError::Missing { key: path.into() })
}

fn opt_u32_at(map: &[(Value, Value)], key: &str, path: &str) -> Result<Option<u32>, MetadataError> {
    match get(map, key) {
        None => Ok(None),
        Some(Value::Integer(n)) => n
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| MetadataError::OutOfRange {
                key: path.into(),
                value: n.to_string(),
            }),
        Some(_) => Err(MetadataError::WrongType {
            key: path.into(),
            expected: "unsigned integer",
        }),
    }
}

fn opt_u32(map: &[(Value, Value)], key: &str) -> Result<Option<u32>, MetadataError> {
    opt_u32_at(map, key, key)
}

fn req_u32(map: &[(Value, Value)], key: &str, path: &str) -> Result<u32, MetadataError> {
    opt_u32_at(map, key, path)?.ok_or_else(|| MetadataError::Missing { key: path.into() })
}

fn opt_flag(map: &[(Value, Value)], key: &str) -> Result<bool, MetadataError> {
    match get(map, key) {
        None => Ok(false),
        Some(Value::Boolean(b)) => Ok(*b),
        Some(Value::Integer(n)) => match n.as_u64() {
            Some(0) => Ok(false),
            Some(1) => Ok(true),
            _ => Err(MetadataError::OutOfRange {
                key: key.into(),
                value: n.to_string(),
            }),
        },
        Some(_) => Err(MetadataError::WrongType {
            key: key.into(),
            expected: "bool or 0/1 integer",
        }),
    }
}

fn decode_exact(raw: &[u8]) -> Result<Value, MetadataError> {
    let mut pos = 0;
    let value = read_value_at(raw, &mut pos)?;
    if pos != raw.len() {
        return Err(MetadataError::Trailing(raw.len() - pos));
    }
    Ok(value)
}

fn read_value_at(buf: &[u8], pos: &mut usize) -> Result<Value, MetadataError> {
    let mut rest = &buf[*pos..];
    let value = rmpv::decode::read_value(&mut rest).map_err(|e| MetadataError::Decode {
        offset: *pos,
        reason: e.to_string(),
    })?;
    *pos = buf.len() - rest.len();
    Ok(value)
}

fn encode_into(out: &mut Vec<u8>, value: &Value) {
    rmpv::encode::write_value(out, value).expect("writing msgpack into a Vec cannot fail");
}

#[derive(Clone, Copy)]
enum Container {
    Map,
    Array,
}

/// Reads a map/array header (the only msgpack framing the document splitter must see).
fn read_container_len(
    buf: &[u8],
    pos: &mut usize,
    kind: Container,
    what: &str,
) -> Result<usize, MetadataError> {
    let at = *pos;
    let wrong = || match kind {
        Container::Map => MetadataError::NotMap { what: what.into() },
        Container::Array => MetadataError::WrongType {
            key: what.into(),
            expected: "array",
        },
    };
    let truncated = || MetadataError::Decode {
        offset: at,
        reason: "truncated container header".into(),
    };
    let marker = *buf.get(at).ok_or_else(truncated)?;
    let (fix, wide16, wide32) = match kind {
        Container::Map => (0x80, 0xde, 0xdf),
        Container::Array => (0x90, 0xdc, 0xdd),
    };
    let (len, used) = match marker {
        m if m & 0xf0 == fix => (usize::from(m & 0x0f), 1),
        m if m == wide16 => (
            usize::from(u16::from_be_bytes(
                buf.get(at + 1..at + 3)
                    .ok_or_else(truncated)?
                    .try_into()
                    .unwrap(),
            )),
            3,
        ),
        m if m == wide32 => (
            u32::from_be_bytes(
                buf.get(at + 1..at + 5)
                    .ok_or_else(truncated)?
                    .try_into()
                    .unwrap(),
            ) as usize,
            5,
        ),
        _ => return Err(wrong()),
    };
    *pos = at + used;
    Ok(len)
}

fn write_container_len(out: &mut Vec<u8>, len: usize, kind: Container) {
    let (fix, wide16, wide32) = match kind {
        Container::Map => (0x80u8, 0xdeu8, 0xdfu8),
        Container::Array => (0x90, 0xdc, 0xdd),
    };
    if len < 16 {
        out.push(fix | len as u8);
    } else if let Ok(n) = u16::try_from(len) {
        out.push(wide16);
        out.extend(n.to_be_bytes());
    } else {
        out.push(wide32);
        out.extend(
            u32::try_from(len)
                .expect("msgpack containers hold at most u32::MAX entries")
                .to_be_bytes(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::SourceCodec;
    use crate::elf::fixtures::{f2_hxaco, kt48_co, KT48_SELECTED};
    use crate::elf::{EnvelopeCodec, KernelImage};
    use peacemaker_ir::envelope::{Envelope, Source};
    use peacemaker_ir::inst::Arch;

    fn kt48_images() -> Vec<KernelImage> {
        Envelope::read(&kt48_co()).unwrap().1
    }
    fn selected() -> KernelImage {
        kt48_images()
            .into_iter()
            .find(|k| k.name == KT48_SELECTED)
            .unwrap()
    }

    #[test]
    fn kt48_selected_metadata_fields_and_bytes() {
        let co = kt48_co();
        let image = selected();
        let meta = &image.metadata.parsed;
        assert_eq!(meta.symbol, format!("{KT48_SELECTED}.kd"));
        assert_eq!(
            (meta.vgpr_count, meta.sgpr_count, meta.wavefront_size),
            (238, 30, 32)
        );
        assert_eq!(
            (meta.max_flat_workgroup_size, meta.kernarg_segment_size),
            (768, 328)
        );
        assert!(meta.workgroup_processor_mode);
        // core.md §0/T5 says "26 .args (11 explicit + hidden)"; pinned llvm-readobj --notes lists
        // 25 for this kernel: 6 global_buffer + 5 by_value explicit, 14 hidden_*.
        assert_eq!(meta.args.len(), 25);
        assert_eq!(
            meta.args
                .iter()
                .filter(|a| !a.value_kind.starts_with("hidden_"))
                .count(),
            11
        );
        // The raw slice is a verbatim run of the .note payload, and re-serialises byte-equal.
        let raw = &image.metadata.raw_msgpack;
        let at = co
            .windows(raw.len())
            .position(|w| w == raw.as_slice())
            .unwrap();
        assert!((0x238..0x238 + 0x1a40).contains(&at), "{at:#x}");
        assert_eq!(serialize_kernel(&image.metadata).unwrap(), *raw);
        check(meta, &image.name, &image.descriptor).unwrap();
    }

    #[test]
    fn every_fixture_kernel_metadata_matches_its_descriptor() {
        let f2 = Source::read(&f2_hxaco(), Arch::Gfx1201).unwrap().1;
        for image in kt48_images().iter().chain(&f2) {
            check(&image.metadata.parsed, &image.name, &image.descriptor)
                .unwrap_or_else(|e| panic!("{e}"));
        }
    }

    #[test]
    fn typed_edits_reserialise_and_keep_unmodelled_keys() {
        let mut meta = selected().metadata;
        meta.parsed.vgpr_count = 256;
        meta.parsed.kernarg_segment_size = 336;
        meta.parsed.uses_dynamic_stack = true;
        meta.parsed.args.push(Kernarg {
            name: "pm_profile".into(),
            size: 8,
            offset: 328,
            value_kind: "global_buffer".into(),
            address_space: Some("global".into()),
        });
        let bytes = serialize_kernel(&meta).unwrap();
        let again = parse_kernel(&bytes).unwrap();
        assert_eq!(again.parsed, meta.parsed);

        let Value::Map(map) = decode_exact(&bytes).unwrap() else {
            panic!("kernel map")
        };
        assert_eq!(
            get(&map, ".language").and_then(Value::as_str),
            Some("OpenCL C")
        );
        assert_eq!(
            get(&map, ".uses_dynamic_stack"),
            Some(&Value::Boolean(true)),
            "bool stays bool"
        );
        assert_eq!(
            get(&map, ".workgroup_processor_mode"),
            Some(&Value::from(1u8)),
            "0/1 integer stays integer"
        );
        let Some(Value::Array(args)) = get(&map, ".args") else {
            panic!(".args")
        };
        let Value::Map(first) = &args[0] else {
            panic!(".args[0]")
        };
        assert_eq!(
            get(first, ".address_space").and_then(Value::as_str),
            Some("global")
        );
        let Value::Map(added) = args.last().unwrap() else {
            panic!("added arg")
        };
        let keys: Vec<_> = added.iter().map(|(k, _)| k.as_str().unwrap()).collect();
        assert_eq!(
            keys,
            [".address_space", ".name", ".offset", ".size", ".value_kind"],
            "keys in LLVM's sorted order"
        );
    }

    #[test]
    fn parse_rejects_non_canonical_and_ill_typed_maps() {
        let raw = selected().metadata.raw_msgpack;
        // `.vgpr_count 238` re-spelled as uint16: same value, bytes a re-encoding cannot reproduce.
        let key = b"\xab.vgpr_count\xcc\xee";
        let at = raw.windows(key.len()).position(|w| w == key).unwrap();
        let wide = [&raw[..at + 12], &[0xcd, 0x00, 0xee], &raw[at + key.len()..]].concat();
        assert_eq!(
            kernel_meta(&decode_exact(&wide).unwrap())
                .unwrap()
                .vgpr_count,
            238
        );
        assert!(matches!(
            parse_kernel(&wide),
            Err(MetadataError::NonCanonical { .. })
        ));

        let edited = |edit: fn(&mut Vec<(Value, Value)>)| {
            let Value::Map(mut map) = decode_exact(&raw).unwrap() else {
                unreachable!()
            };
            edit(&mut map);
            let mut out = Vec::new();
            encode_into(&mut out, &Value::Map(map));
            parse_kernel(&out).unwrap_err()
        };
        assert_eq!(
            edited(|m| m.retain(|(k, _)| k.as_str() != Some(".symbol"))),
            MetadataError::Missing {
                key: ".symbol".into()
            }
        );
        assert_eq!(
            edited(|m| *get_mut(m, ".wavefront_size").unwrap() = Value::from("32")),
            MetadataError::WrongType {
                key: ".wavefront_size".into(),
                expected: "unsigned integer"
            }
        );
        assert_eq!(
            edited(|m| *get_mut(m, ".workgroup_processor_mode").unwrap() = Value::from(2u8)),
            MetadataError::OutOfRange {
                key: ".workgroup_processor_mode".into(),
                value: "2".into()
            }
        );
        assert_eq!(
            parse_kernel(&[raw.as_slice(), &[0xc0]].concat()).unwrap_err(),
            MetadataError::Trailing(1)
        );

        let mut doc = Vec::new();
        encode_into(
            &mut doc,
            &Value::Map(vec![(
                Value::from("amdhsa.version"),
                Value::Array(vec![Value::from(1u8), Value::from(2u8)]),
            )]),
        );
        assert_eq!(
            MetadataDoc::parse(&doc).unwrap_err(),
            MetadataError::NoKernels
        );
    }

    #[test]
    fn check_rejects_metadata_that_contradicts_the_descriptor() {
        let image = selected();
        let with = |edit: fn(&mut KernelMeta)| {
            let mut meta = image.metadata.parsed.clone();
            edit(&mut meta);
            check(&meta, &image.name, &image.descriptor)
        };
        // The descriptor reserves 240 VGPRs, though smaller kernels are legal.
        assert_eq!(with(|m| m.vgpr_count = 27), Ok(()));
        assert_eq!(with(|m| m.vgpr_count = 240), Ok(()));
        assert!(matches!(
            with(|m| m.vgpr_count = 241),
            Err(MetadataError::Inconsistent { .. })
        ));
        assert!(matches!(
            with(|m| m.private_segment_fixed_size = 16),
            Err(MetadataError::Inconsistent { .. })
        ));
        assert!(matches!(
            with(|m| m.wavefront_size = 64),
            Err(MetadataError::Inconsistent { .. })
        ));
        assert!(matches!(
            with(|m| m.symbol = "other.kd".into()),
            Err(MetadataError::Inconsistent { .. })
        ));
    }
}
