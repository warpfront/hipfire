//! C2: HIP offload bundle read/write, and the `Source` (ELF or bundled ELF) codec.
//!
//! Uncompressed `clang-offload-bundler` format: the 24-byte magic, a little-endian `u64`
//! entry count, then per entry `u64` offset (absolute), `u64` size, `u64` triple length and
//! the triple bytes. Payloads are not owned by the `Bundle`: `read` borrows them from the
//! input and `write` takes them back (the device payload is the re-emitted ELF), so a bundle
//! round trip is header + payloads + the retained padding `Gap`s.

use peacemaker_ir::envelope::{Bundle, BundleEntry, Envelope, Source};
use peacemaker_ir::inst::Arch;

use crate::elf::{assemble, find_gaps, Chunk, ElfError, EnvelopeCodec, KernelImage, KernelParts, Overlap, Region};

pub const MAGIC: &[u8; 24] = b"__CLANG_OFFLOAD_BUNDLE__";
/// Magic of the compressed bundle format (`--offload-compress`), which is not handled.
pub const COMPRESSED_MAGIC: &[u8; 4] = b"CCOB";
/// Triple prefix of the HIP device entry (`hipv4-amdgcn-amd-amdhsa--<arch>[:features]`).
pub const HIP_DEVICE_TRIPLE_PREFIX: &str = "hipv4-amdgcn-amd-amdhsa--";

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum BundleError {
    #[error("not an uncompressed clang offload bundle (missing __CLANG_OFFLOAD_BUNDLE__ magic)")]
    NotBundle,
    #[error("compressed offload bundles (CCOB) are not supported; build with --no-offload-compress")]
    Compressed,
    #[error("bundle header is truncated at byte {0}")]
    Truncated(u64),
    #[error("entry {index} triple is not UTF-8")]
    Triple { index: usize },
    #[error("entry {index} ({triple}) payload [{start:#x}, {end:#x}) lies outside the {len}-byte file")]
    Payload { index: usize, triple: String, start: u64, end: u64, len: u64 },
    #[error("{first} and {second} overlap at byte {offset:#x}")]
    Overlap { first: String, second: String, offset: u64 },
    #[error("write needs one payload per entry: {expected} entries, {got} payloads")]
    PayloadCount { expected: usize, got: usize },
    #[error("no bundle entry for {0}")]
    NoEntry(String),
    #[error("{count} bundle entries match {triple}")]
    Ambiguous { count: usize, triple: String },
}

impl From<Overlap> for BundleError {
    fn from(o: Overlap) -> Self { BundleError::Overlap { first: o.first, second: o.second, offset: o.offset } }
}

/// Bundle read/write (`Bundle::read(..)` with this trait in scope).
pub trait BundleCodec: Sized {
    fn is_bundle(bytes: &[u8]) -> bool;
    /// Splits a bundle into its structure and its payloads (one per entry, in header order).
    fn read(bytes: &[u8]) -> Result<(Self, Vec<&[u8]>), BundleError>;
    /// Emits the bundle with `payloads` (header order); entry sizes come from the payloads.
    fn write(&self, payloads: &[&[u8]]) -> Result<Vec<u8>, BundleError>;
    /// Index of the single HIP device entry for `arch` (`hipv4-amdgcn-amd-amdhsa--<arch>`,
    /// optionally followed by `:feature` suffixes).
    fn device_entry(&self, arch: Arch) -> Result<usize, BundleError>;
}

impl BundleCodec for Bundle {
    fn is_bundle(bytes: &[u8]) -> bool { bytes.starts_with(MAGIC) }

    fn read(bytes: &[u8]) -> Result<(Bundle, Vec<&[u8]>), BundleError> {
        if bytes.starts_with(COMPRESSED_MAGIC) {
            return Err(BundleError::Compressed);
        }
        if !Self::is_bundle(bytes) {
            return Err(BundleError::NotBundle);
        }
        let mut pos = MAGIC.len();
        let u64_at = |pos: &mut usize| -> Result<u64, BundleError> {
            let word = bytes.get(*pos..*pos + 8).ok_or(BundleError::Truncated(*pos as u64))?;
            *pos += 8;
            Ok(u64::from_le_bytes(word.try_into().unwrap()))
        };
        let count = u64_at(&mut pos)?;
        // Every entry needs at least 24 header bytes; bounds the allocation below.
        if count > (bytes.len() - pos) as u64 / 24 {
            return Err(BundleError::Truncated(bytes.len() as u64));
        }
        let mut entries = Vec::with_capacity(count as usize);
        let mut sizes = Vec::with_capacity(count as usize);
        for index in 0..count as usize {
            let offset = u64_at(&mut pos)?;
            let size = u64_at(&mut pos)?;
            let triple_len = u64_at(&mut pos)?;
            let triple = usize::try_from(triple_len).ok().and_then(|n| bytes.get(pos..pos.checked_add(n)?))
                .ok_or(BundleError::Truncated(pos as u64))?;
            pos += triple.len();
            let triple = String::from_utf8(triple.to_vec()).map_err(|_| BundleError::Triple { index })?;
            let end = offset.checked_add(size).filter(|&end| end <= bytes.len() as u64)
                .ok_or_else(|| BundleError::Payload { index, triple: triple.clone(), start: offset, end: offset.saturating_add(size), len: bytes.len() as u64 })?;
            sizes.push((offset, end));
            entries.push(BundleEntry { triple, offset });
        }
        let mut regions = vec![Region::new("bundle header", 0, pos as u64)];
        regions.extend(entries.iter().zip(&sizes).map(|(e, &(start, end))| Region::new(format!("payload {}", e.triple), start, end - start)));
        let gaps = find_gaps(bytes, regions)?;
        let payloads = sizes.iter().map(|&(start, end)| &bytes[start as usize..end as usize]).collect();
        Ok((Bundle { entries, gaps }, payloads))
    }

    fn write(&self, payloads: &[&[u8]]) -> Result<Vec<u8>, BundleError> {
        if payloads.len() != self.entries.len() {
            return Err(BundleError::PayloadCount { expected: self.entries.len(), got: payloads.len() });
        }
        let mut header = MAGIC.to_vec();
        header.extend((self.entries.len() as u64).to_le_bytes());
        for (entry, payload) in self.entries.iter().zip(payloads) {
            header.extend(entry.offset.to_le_bytes());
            header.extend((payload.len() as u64).to_le_bytes());
            header.extend((entry.triple.len() as u64).to_le_bytes());
            header.extend(entry.triple.as_bytes());
        }
        let mut chunks = vec![Chunk::bytes("bundle header", 0, header)];
        chunks.extend(self.entries.iter().zip(payloads).map(|(e, p)| Chunk::bytes(format!("payload {}", e.triple), e.offset, *p)));
        chunks.extend(self.gaps.iter().map(Chunk::gap));
        Ok(assemble(chunks)?)
    }

    fn device_entry(&self, arch: Arch) -> Result<usize, BundleError> {
        let triple = format!("{HIP_DEVICE_TRIPLE_PREFIX}{}", arch_name(arch));
        let matches: Vec<usize> = self.entries.iter().enumerate()
            .filter(|(_, e)| e.triple.strip_prefix(triple.as_str()).is_some_and(|rest| rest.is_empty() || rest.starts_with(':')))
            .map(|(i, _)| i)
            .collect();
        match matches[..] {
            [index] => Ok(index),
            [] => Err(BundleError::NoEntry(triple)),
            _ => Err(BundleError::Ambiguous { count: matches.len(), triple }),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error(transparent)]
    Bundle(#[from] BundleError),
    #[error(transparent)]
    Elf(#[from] ElfError),
    #[error("bundle entry {triple} carries a {len}-byte payload that `Source` cannot hold; only the device ELF may be non-empty")]
    ForeignPayload { triple: String, len: usize },
}

/// Reads and writes a lifter input: a code object ELF, or a HIP bundle whose `arch` device
/// entry is that ELF (`Source::read(..)` with this trait in scope).
pub trait SourceCodec: Sized {
    fn read(bytes: &[u8], arch: Arch) -> Result<(Self, Vec<KernelImage>), SourceError>;
    fn write(&self, parts: &[KernelParts<'_>]) -> Result<Vec<u8>, SourceError>;
}

impl SourceCodec for Source {
    fn read(bytes: &[u8], arch: Arch) -> Result<(Source, Vec<KernelImage>), SourceError> {
        if !Bundle::is_bundle(bytes) && !bytes.starts_with(COMPRESSED_MAGIC) {
            let (elf, images) = Envelope::read(bytes)?;
            return Ok((Source { elf, bundle: None }, images));
        }
        let (bundle, payloads) = Bundle::read(bytes)?;
        let device = bundle.device_entry(arch)?;
        sole_device_entry(&bundle)?;
        if let Some((entry, payload)) = bundle.entries.iter().zip(&payloads).enumerate()
            .find_map(|(i, pair)| (i != device && !pair.1.is_empty()).then_some(pair))
        {
            return Err(SourceError::ForeignPayload { triple: entry.triple.clone(), len: payload.len() });
        }
        let (elf, images) = Envelope::read(payloads[device])?;
        Ok((Source { elf, bundle: Some(bundle) }, images))
    }

    fn write(&self, parts: &[KernelParts<'_>]) -> Result<Vec<u8>, SourceError> {
        let elf = self.elf.write(parts)?;
        let Some(bundle) = &self.bundle else { return Ok(elf) };
        let device = sole_device_entry(bundle)?;
        let mut payloads: Vec<&[u8]> = vec![&[]; bundle.entries.len()];
        payloads[device] = &elf;
        Ok(bundle.write(&payloads)?)
    }
}

/// `Source` keeps no payloads, so its bundle must have exactly one HIP device entry: the one
/// the ELF is written back into.
fn sole_device_entry(bundle: &Bundle) -> Result<usize, BundleError> {
    let devices: Vec<usize> = bundle.entries.iter().enumerate()
        .filter(|(_, e)| e.triple.starts_with(HIP_DEVICE_TRIPLE_PREFIX))
        .map(|(i, _)| i)
        .collect();
    match devices[..] {
        [index] => Ok(index),
        [] => Err(BundleError::NoEntry(HIP_DEVICE_TRIPLE_PREFIX.into())),
        _ => Err(BundleError::Ambiguous { count: devices.len(), triple: HIP_DEVICE_TRIPLE_PREFIX.into() }),
    }
}

fn arch_name(arch: Arch) -> &'static str {
    match arch {
        Arch::Gfx1010 => "gfx1010",
        Arch::Gfx1030 => "gfx1030",
        Arch::Gfx1100 => "gfx1100",
        Arch::Gfx1151 => "gfx1151",
        Arch::Gfx1201 => "gfx1201",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elf::fixtures::f2_hxaco;

    #[test]
    fn f2_entries_and_device_payload() {
        let hxaco = f2_hxaco();
        let (bundle, payloads) = Bundle::read(&hxaco).unwrap();
        let triples: Vec<_> = bundle.entries.iter().map(|e| e.triple.as_str()).collect();
        assert_eq!(triples, ["host-x86_64-unknown-linux-gnu-", "hipv4-amdgcn-amd-amdhsa--gfx1201"]);
        assert_eq!(bundle.entries.iter().map(|e| e.offset).collect::<Vec<_>>(), [0x1000, 0x1000]);
        let device = bundle.device_entry(Arch::Gfx1201).unwrap();
        assert_eq!((payloads[0].len(), payloads[device].len()), (0, 0x57640));
        assert!(payloads[device].starts_with(b"\x7fELF"));
        assert_eq!(bundle.write(&payloads).unwrap(), hxaco);
    }

    #[test]
    fn device_payload_resizes_in_place() {
        let hxaco = f2_hxaco();
        let (bundle, payloads) = Bundle::read(&hxaco).unwrap();
        let device = bundle.device_entry(Arch::Gfx1201).unwrap();
        let grown = [payloads[device], &[0xaa; 8]].concat();
        let mut out = payloads.clone();
        out[device] = &grown;
        let bytes = bundle.write(&out).unwrap();
        assert_eq!(bytes.len(), hxaco.len() + 8);
        let (again, again_payloads) = Bundle::read(&bytes).unwrap();
        assert_eq!(again, bundle);
        assert_eq!(again_payloads[device], grown.as_slice());
    }

    #[test]
    fn device_entry_matches_arch_and_feature_suffix_only() {
        let entry = |triple: &str| Bundle { entries: vec![BundleEntry { triple: triple.into(), offset: 0x1000 }], gaps: Vec::new() };
        assert_eq!(entry("hipv4-amdgcn-amd-amdhsa--gfx1201:xnack-").device_entry(Arch::Gfx1201), Ok(0));
        assert!(matches!(entry("hipv4-amdgcn-amd-amdhsa--gfx12010").device_entry(Arch::Gfx1201), Err(BundleError::NoEntry(_))));
        assert_eq!(entry("hipv4-amdgcn-amd-amdhsa--gfx1201").device_entry(Arch::Gfx1100),
            Err(BundleError::NoEntry("hipv4-amdgcn-amd-amdhsa--gfx1100".into())));
        let mut two = entry("hipv4-amdgcn-amd-amdhsa--gfx1201");
        two.entries.push(BundleEntry { triple: "hipv4-amdgcn-amd-amdhsa--gfx1201:sramecc-".into(), offset: 0x2000 });
        assert!(matches!(two.device_entry(Arch::Gfx1201), Err(BundleError::Ambiguous { count: 2, .. })));
    }

    #[test]
    fn rejects_what_it_cannot_represent() {
        assert_eq!(Bundle::read(b"CCOB\x02\x00").unwrap_err(), BundleError::Compressed);
        assert_eq!(Bundle::read(b"\x7fELF\x02\x01\x01").unwrap_err(), BundleError::NotBundle);
        let header = |count: u64, entry: &[u64]| {
            let mut bytes = MAGIC.to_vec();
            bytes.extend(count.to_le_bytes());
            entry.iter().for_each(|w| bytes.extend(w.to_le_bytes()));
            bytes
        };
        assert!(matches!(Bundle::read(&header(1, &[0x40, 0])), Err(BundleError::Truncated(_))));
        let mut past_end = header(1, &[0x40, 0x100, 1]);
        past_end.push(b'x');
        assert!(matches!(Bundle::read(&past_end), Err(BundleError::Payload { index: 0, .. })));

        let hxaco = f2_hxaco();
        let (bundle, payloads) = Bundle::read(&hxaco).unwrap();
        assert_eq!(bundle.write(&payloads[..1]).unwrap_err(), BundleError::PayloadCount { expected: 2, got: 1 });
    }

    /// `Source` keeps only the device ELF, so bundles carrying anything else are rejected
    /// rather than silently losing bytes on re-emission.
    #[test]
    fn source_rejects_bundles_it_cannot_re_emit() {
        let hxaco = f2_hxaco();
        let (_, payloads) = Bundle::read(&hxaco).unwrap();
        let elf = payloads[1];
        let entry = |triple: &str, offset| BundleEntry { triple: triple.into(), offset };
        let host = entry("host-x86_64-unknown-linux-gnu-", 0x100);
        let device = entry("hipv4-amdgcn-amd-amdhsa--gfx1201", 0x1000);

        let with_host = Bundle { entries: vec![host.clone(), device.clone()], gaps: Vec::new() }.write(&[b"host", elf]).unwrap();
        assert!(matches!(Source::read(&with_host, Arch::Gfx1201),
            Err(SourceError::ForeignPayload { triple, len: 4 }) if triple.starts_with("host-")));

        let second = entry("hipv4-amdgcn-amd-amdhsa--gfx1100", 0x100);
        let two_devices = Bundle { entries: vec![second, device], gaps: Vec::new() }.write(&[&[], elf]).unwrap();
        assert!(matches!(Source::read(&two_devices, Arch::Gfx1201), Err(SourceError::Bundle(BundleError::Ambiguous { count: 2, .. }))));
    }
}
