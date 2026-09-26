//! CPU-only packet decoding for the default, single-queue gfx1201 IB subset.
//!
//! This validates packet shapes, not queue state, register dependencies, code
//! addresses, or a certified tape. S1 must perform those independent checks.

use crate::Pm4BuildError;
use std::fmt;

const SET_SH_REG: u32 = 0x76;
const DISPATCH_DIRECT: u32 = 0x15;
const EVENT_WRITE: u32 = 0x46;
const ACQUIRE_MEM: u32 = 0x58;
const SH_FIRST: u32 = 0x200;
const SH_LAST: u32 = 0x2ff;
const INITIATOR_BASE: u32 = 0x25;
const WAVE32: u32 = 0x8000;
const SYSTEM_GCR: u32 = 0xc3b1;
const INTER_NODE_GCR: u32 = 0x10180;
const ACQUIRE_PREFIX: [u32; 6] = [0, u32::MAX, 0x00ff_ffff, 0, 0, 0xa];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Pm4Family {
    Gfx10,
    Gfx11,
    Gfx12_0,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParseMode {
    Certified,
    Diagnostic,
}

/// S0 has no absolute-address packets. This empty table is intentionally not
/// an implicit license to accept addresses; S1 must resolve them explicitly.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResolvedRelocations;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShReg(u16);

impl ShReg {
    pub fn try_new(offset: u32) -> Option<Self> {
        (SH_FIRST..=SH_LAST).contains(&offset).then_some(Self(offset as u16))
    }

    pub fn offset(self) -> u32 {
        u32::from(self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NonZeroGrid([u32; 3]);

impl NonZeroGrid {
    pub fn try_new(dimensions: [u32; 3]) -> Option<Self> {
        dimensions.iter().all(|&n| n != 0).then_some(Self(dimensions))
    }

    pub fn dimensions(self) -> [u32; 3] {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DispatchInitiator(u32);

impl DispatchInitiator {
    pub fn try_new(word: u32) -> Option<Self> {
        (word == INITIATOR_BASE || word == INITIATOR_BASE | WAVE32).then_some(Self(word))
    }

    pub fn word(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheScope {
    System,
    InterNode,
}

impl CacheScope {
    fn gcr(self) -> u32 {
        match self {
            Self::System => SYSTEM_GCR,
            Self::InterNode => INTER_NODE_GCR,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Pm4Packet {
    SetShRegs { first: ShReg, values: Vec<u32> },
    DispatchDirect {
        dimensions: NonZeroGrid,
        initiator: DispatchInitiator,
    },
    EventComputeIdle,
    Acquire { scope: CacheScope, gcr: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Pm4ParseError {
    pub dword: usize,
    pub reason: &'static str,
}

impl fmt::Display for Pm4ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PM4 dword {}: {}", self.dword, self.reason)
    }
}

impl std::error::Error for Pm4ParseError {}

fn reject(dword: usize, reason: &'static str) -> Pm4ParseError {
    Pm4ParseError { dword, reason }
}

/// Decode only the default single-queue gfx1201 packet shapes. Even in
/// diagnostic mode, timestamp and other opcodes remain unsupported in S0.
/// The caller must separately establish the ASIC is gfx1201 (not gfx1200).
pub fn parse_pm4(
    family: Pm4Family,
    words: &[u32],
    _mode: ParseMode,
    _reloc: &ResolvedRelocations,
) -> Result<Vec<Pm4Packet>, Pm4ParseError> {
    if family != Pm4Family::Gfx12_0 {
        return Err(reject(0, "unsupported PM4 family"));
    }
    let mut packets = Vec::new();
    let mut at = 0;
    while at < words.len() {
        let header = words[at];
        if header >> 30 != 3 {
            return Err(reject(at, "not a PACKET3 header"));
        }
        let count = ((header >> 16) & 0x3fff) as usize + 1;
        let opcode = (header >> 8) & 0xff;
        let compute = match opcode {
            SET_SH_REG | DISPATCH_DIRECT => 2,
            EVENT_WRITE | ACQUIRE_MEM => 0,
            _ => return Err(reject(at, "unsupported PM4 opcode (including diagnostics)")),
        };
        if header & 0xff != compute {
            return Err(reject(at, "reserved header bits or wrong compute bit"));
        }
        let end = at
            .checked_add(1 + count)
            .filter(|&end| end <= words.len())
            .ok_or_else(|| reject(at, "truncated packet"))?;
        let body = &words[at + 1..end];
        let packet = match opcode {
            SET_SH_REG => {
                if count < 2 {
                    return Err(reject(at, "SET_SH_REG needs a register and values"));
                }
                let first = ShReg::try_new(body[0])
                    .ok_or_else(|| reject(at + 1, "register outside gfx12 SH window"))?;
                if first.offset() + (count - 2) as u32 > SH_LAST {
                    return Err(reject(at, "register run exceeds gfx12 SH window"));
                }
                Pm4Packet::SetShRegs {
                    first,
                    values: body[1..].to_vec(),
                }
            }
            DISPATCH_DIRECT => {
                if count != 4 {
                    return Err(reject(at, "DISPATCH_DIRECT requires four body dwords"));
                }
                let dimensions = NonZeroGrid::try_new([body[0], body[1], body[2]])
                    .ok_or_else(|| reject(at, "zero dispatch dimension"))?;
                let initiator = DispatchInitiator::try_new(body[3])
                    .ok_or_else(|| reject(at + 4, "unsupported dispatch initiator"))?;
                Pm4Packet::DispatchDirect {
                    dimensions,
                    initiator,
                }
            }
            EVENT_WRITE => {
                if body != [0x407] {
                    return Err(reject(at, "not a CS_PARTIAL_FLUSH event"));
                }
                Pm4Packet::EventComputeIdle
            }
            ACQUIRE_MEM => {
                if count != 7 || body[..6] != ACQUIRE_PREFIX {
                    return Err(reject(at, "unsupported ACQUIRE_MEM shape"));
                }
                let scope = match body[6] {
                    SYSTEM_GCR => CacheScope::System,
                    INTER_NODE_GCR => CacheScope::InterNode,
                    _ => return Err(reject(at + 7, "unsupported ACQUIRE_MEM scope")),
                };
                Pm4Packet::Acquire {
                    scope,
                    gcr: body[6],
                }
            }
            _ => unreachable!("opcode checked before body decode"),
        };
        packets.push(packet);
        at = end;
    }
    Ok(packets)
}

fn header(opcode: u32, count: usize, compute: bool) -> u32 {
    (3 << 30) | ((count as u32 - 1) << 16) | (opcode << 8) | (u32::from(compute) << 1)
}

/// Encode only S0 packet shapes; malformed hand-built variants fail closed.
/// The existing builder error's `MalformedStream` identifies the packet index
/// when an invalid typed packet cannot be emitted.
pub fn encode_pm4(
    family: Pm4Family,
    packets: &[Pm4Packet],
    _reloc: &ResolvedRelocations,
) -> Result<Vec<u32>, Pm4BuildError> {
    if family != Pm4Family::Gfx12_0 {
        return Err(Pm4BuildError::MalformedStream { dword: 0 });
    }
    let mut words = Vec::new();
    for packet in packets {
        let invalid = || Pm4BuildError::MalformedStream { dword: words.len() };
        match packet {
            Pm4Packet::SetShRegs { first, values } => {
                if values.is_empty() || values.len() > (SH_LAST - first.offset() + 1) as usize {
                    return Err(invalid());
                }
                words.push(header(SET_SH_REG, 1 + values.len(), true));
                words.push(first.offset());
                words.extend_from_slice(values);
            }
            Pm4Packet::DispatchDirect {
                dimensions,
                initiator,
            } => {
                words.push(header(DISPATCH_DIRECT, 4, true));
                words.extend_from_slice(&dimensions.dimensions());
                words.push(initiator.word());
            }
            Pm4Packet::EventComputeIdle => {
                words.extend_from_slice(&[header(EVENT_WRITE, 1, false), 0x407]);
            }
            Pm4Packet::Acquire { scope, gcr } => {
                if *gcr != scope.gcr() {
                    return Err(invalid());
                }
                words.push(header(ACQUIRE_MEM, 7, false));
                words.extend_from_slice(&ACQUIRE_PREFIX);
                words.push(*gcr);
            }
        }
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gfx12Pm4CommandBuffer;

    fn decode(words: &[u32]) -> Result<Vec<Pm4Packet>, Pm4ParseError> {
        parse_pm4(
            Pm4Family::Gfx12_0,
            words,
            ParseMode::Certified,
            &ResolvedRelocations,
        )
    }

    // The existing builder generates the ownership/acquire/completion packets.
    // Its dispatch API requires an HSA-loaded Kernel, so a CPU-only fixture
    // inserts the exact SET_SH_REG and DISPATCH_DIRECT packet shapes it emits.
    fn fixture() -> (Vec<u32>, usize, usize) {
        let mut builder = Gfx12Pm4CommandBuffer::new();
        builder.acquire_system_gfx12();
        let mut words = builder.dwords().to_vec();
        let set_at = words.len();
        words.extend_from_slice(&[header(SET_SH_REG, 3, true), 0x20c, 0x1234, 0x5678]);
        let dispatch_at = words.len();
        words.extend_from_slice(&[header(DISPATCH_DIRECT, 4, true), 256, 2, 1, 0x8025]);
        builder.wait_compute_idle();
        builder.acquire_inter_node_gfx12();
        words.extend_from_slice(&builder.dwords()[8..]);
        (words, set_at, dispatch_at)
    }

    #[test]
    fn pm4_decode_builder_stream_roundtrips_byte_for_byte() {
        let mut builder = Gfx12Pm4CommandBuffer::new();
        builder.acquire_system_gfx12();
        builder.wait_compute_idle();
        builder.acquire_inter_node_gfx12();
        let builder_packets = decode(builder.dwords()).unwrap();
        assert_eq!(
            encode_pm4(Pm4Family::Gfx12_0, &builder_packets, &ResolvedRelocations).unwrap(),
            builder.dwords()
        );
        let (words, _, _) = fixture();
        let packets = decode(&words).unwrap();
        assert_eq!(packets.len(), 5);
        assert_eq!(
            packets[0],
            Pm4Packet::Acquire {
                scope: CacheScope::System,
                gcr: SYSTEM_GCR,
            }
        );
        assert_eq!(
            packets[2],
            Pm4Packet::DispatchDirect {
                dimensions: NonZeroGrid::try_new([256, 2, 1]).unwrap(),
                initiator: DispatchInitiator::try_new(0x8025).unwrap(),
            }
        );
        let encoded = encode_pm4(Pm4Family::Gfx12_0, &packets, &ResolvedRelocations).unwrap();
        assert_eq!(encoded, words);
        let bytes = |words: &[u32]| words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>();
        assert_eq!(bytes(&encoded), bytes(&words));
    }

    #[test]
    fn pm4_decode_rejects_corrupt_headers_lengths_and_registers() {
        let (words, set_at, dispatch_at) = fixture();
        let mut bad = words.clone();
        bad[set_at] += 1 << 16; // count eats the next packet's header
        assert!(decode(&bad).is_err());
        let mut bad = words.clone();
        bad[set_at] = header(EVENT_WRITE, 3, false); // known opcode, wrong length
        assert!(decode(&bad).is_err());
        let mut bad = words.clone();
        bad[set_at + 1] = 0x300;
        assert!(decode(&bad).is_err());
        let mut bad = words.clone();
        bad[set_at + 1] = 0x2ff; // two registers cross the window end
        assert!(decode(&bad).is_err());
        let mut bad = words.clone();
        bad[dispatch_at] = header(DISPATCH_DIRECT, 3, true);
        assert!(decode(&bad).is_err());
        let mut bad = words.clone();
        bad[dispatch_at + 1] = 0;
        assert!(decode(&bad).is_err());
        let mut bad = words.clone();
        bad[dispatch_at + 4] = 0x8027; // reserved initiator bit
        assert!(decode(&bad).is_err());
        let mut bad = words.clone();
        bad[set_at] &= !2; // graphics instead of compute
        assert!(decode(&bad).is_err());
        let mut bad = words.clone();
        bad[0] &= !(3 << 30);
        assert!(decode(&bad).is_err());
        let mut bad = words.clone();
        bad[3] = 0; // noncanonical acquire prefix
        assert!(decode(&bad).is_err());
        let mut bad = words.clone();
        bad[7] = 0; // nondefault acquire policy
        assert!(decode(&bad).is_err());
        let mut bad = words.clone();
        bad[dispatch_at] = header(0x40, 4, true); // diagnostic COPY_DATA
        assert!(decode(&bad).is_err());
        let mut bad = words.clone();
        bad[dispatch_at] = header(0x7f, 4, true);
        assert!(decode(&bad).is_err());
        for end in [1, set_at + 1, dispatch_at + 2, words.len() - 1] {
            assert!(decode(&words[..end]).is_err(), "accepted truncated stream {end}");
        }
    }

    #[test]
    fn pm4_decode_rejects_unsupported_families_diagnostics_and_invalid_encoding() {
        let (words, _, _) = fixture();
        for family in [Pm4Family::Gfx10, Pm4Family::Gfx11] {
            assert!(parse_pm4(family, &words, ParseMode::Certified, &ResolvedRelocations).is_err());
            assert!(encode_pm4(family, &[], &ResolvedRelocations).is_err());
        }
        let mut builder = Gfx12Pm4CommandBuffer::new();
        builder.wait_compute_idle();
        let timed = builder.with_gpu_timestamps(0x1000, 0x2000);
        assert!(parse_pm4(
            Pm4Family::Gfx12_0,
            timed.dwords(),
            ParseMode::Diagnostic,
            &ResolvedRelocations
        )
        .is_err());
        assert!(encode_pm4(
            Pm4Family::Gfx12_0,
            &[Pm4Packet::SetShRegs {
                first: ShReg::try_new(0x200).unwrap(),
                values: vec![],
            }],
            &ResolvedRelocations
        )
        .is_err());
        assert!(encode_pm4(
            Pm4Family::Gfx12_0,
            &[Pm4Packet::Acquire {
                scope: CacheScope::System,
                gcr: INTER_NODE_GCR,
            }],
            &ResolvedRelocations
        )
        .is_err());
    }
}
