// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! CDO (configuration data object) as found inside an NPU PDI partition.
//!
//! Format (bootgen `cdo-driver/cdo_driver.c`, `utils/src/cdo-binary.c`): a 5-word header
//! `[4, "CDO\0", 0x200, body_words, ~sum(previous 4)]`, then commands. A command header word is
//! `len << 16 | module << 8 | id` (len = payload words; len 255 = long form, next word is length).
//! Inside PDIs bootgen narrows the raw 64-bit commands to the 32-bit forms used here:
//! WRITE 0x103 `[addr, val]`, MASK_WRITE 0x102 `[addr, mask, val]`,
//! DMA_WRITE 0x105 `[addr_hi, addr_lo, data...]`, NOP 0x111 `[zeros]`.
//! DMA_WRITE payloads are kept 16-byte aligned with NOPs (as bootgen does).

pub const CMD_MASK_WRITE: u32 = 0x102;
pub const CMD_WRITE: u32 = 0x103;
pub const CMD_DMA_WRITE: u32 = 0x105;
pub const CMD_NOP: u32 = 0x111;

#[derive(Default, Clone)]
pub struct Cdo {
    body: Vec<u32>,
}

impl Cdo {
    pub fn new() -> Cdo {
        Cdo::default()
    }

    fn header(&mut self, id: u32, len: usize) {
        if len < 255 {
            self.body.push(((len as u32) << 16) | id);
        } else {
            self.body.push((255 << 16) | id);
            self.body.push(len as u32);
        }
    }

    pub fn write(&mut self, addr: u32, val: u32) -> &mut Self {
        self.header(CMD_WRITE, 2);
        self.body.extend([addr, val]);
        self
    }

    pub fn mask_write(&mut self, addr: u32, mask: u32, val: u32) -> &mut Self {
        self.header(CMD_MASK_WRITE, 3);
        self.body.extend([addr, mask, val]);
        self
    }

    pub fn nop(&mut self, payload_words: usize) -> &mut Self {
        self.header(CMD_NOP, payload_words);
        self.body.extend(std::iter::repeat(0).take(payload_words));
        self
    }

    /// Block write; the data payload is placed at a 16-byte boundary of the CDO.
    pub fn dma_write(&mut self, addr: u32, data: &[u32]) -> &mut Self {
        let hdr_words = if data.len() + 2 < 255 { 1 } else { 2 };
        // CDO starts after the 5-word header; payload offset = 5 + body + hdr + 2 (address words)
        let pad = (4 - (5 + self.body.len() + hdr_words + 2) % 4) % 4;
        if pad > 0 {
            self.nop(pad - 1);
        }
        self.header(CMD_DMA_WRITE, data.len() + 2);
        self.body.extend([0, addr]);
        self.body.extend_from_slice(data);
        self
    }

    pub fn body_words(&self) -> &[u32] {
        &self.body
    }

    /// Construct from already-encoded command words (e.g. lifted from an existing PDI).
    pub fn from_body(body: Vec<u32>) -> Cdo {
        Cdo { body }
    }

    pub fn to_words(&self) -> Vec<u32> {
        let h = [4u32, 0x004F_4443, 0x0000_0200, self.body.len() as u32];
        let ck = !h.iter().fold(0u32, |a, &w| a.wrapping_add(w));
        let mut out = Vec::with_capacity(5 + self.body.len());
        out.extend(h);
        out.push(ck);
        out.extend_from_slice(&self.body);
        out
    }
}

/// Decoded command for inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cmd {
    Write(u32, u32),
    MaskWrite(u32, u32, u32),
    DmaWrite(u64, Vec<u32>),
    Nop(usize),
    Other(u32, Vec<u32>),
}

/// Parse a full CDO (header + body). Returns None on a malformed header.
pub fn parse(words: &[u32]) -> Option<Vec<Cmd>> {
    if words.len() < 5 || words[0] != 4 || words[1] != 0x004F_4443 {
        return None;
    }
    let n = words[3] as usize;
    let body = words.get(5..5 + n)?;
    let mut i = 0;
    let mut out = Vec::new();
    while i < body.len() {
        let h = body[i];
        i += 1;
        let mut len = (h >> 16) as usize;
        if len == 255 {
            len = body[i] as usize;
            i += 1;
        }
        let p = body.get(i..i + len)?;
        i += len;
        out.push(match h & 0xffff {
            CMD_WRITE => Cmd::Write(p[0], p[1]),
            CMD_MASK_WRITE => Cmd::MaskWrite(p[0], p[1], p[2]),
            CMD_DMA_WRITE => Cmd::DmaWrite(((p[0] as u64) << 32) | p[1] as u64, p[2..].to_vec()),
            CMD_NOP => Cmd::Nop(len),
            id => Cmd::Other(id, p.to_vec()),
        });
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn header_checksum_and_alignment() {
        let mut c = Cdo::new();
        c.mask_write(0x0023_2000, 1, 0)
            .dma_write(0x0022_0000, &[1, 2, 3]);
        let w = c.to_words();
        assert_eq!(w[4], !(4u32 + 0x004F_4443 + 0x200 + w[3]));
        // payload of the DMA write must start at a 4-word boundary
        let pos = w.iter().position(|&x| x & 0xffff == CMD_DMA_WRITE).unwrap();
        assert_eq!((pos + 3) % 4, 0);
        let cmds = parse(&w).unwrap();
        assert_eq!(
            cmds.last().unwrap(),
            &Cmd::DmaWrite(0x0022_0000, vec![1, 2, 3])
        );
    }
}
