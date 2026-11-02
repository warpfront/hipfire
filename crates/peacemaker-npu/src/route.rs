// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! AIE2P circuit and packet stream-switch routes.
//!
//! Port tables and register fields transcribed from AMD/Xilinx aie-rt
//! `xaie2pgbl_params.h` / `xaie_ss.c` (MIT). Port indices are asymmetric:
//! north has six masters/four slaves, south four masters/six slaves on compute
//! and memory tiles. Shim DMA connects through the NOC mux, not a switch DMA port.

use crate::{
    cdo::Cdo,
    dma::{Location, TileType},
    txn::Txn,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Port {
    Core(u8),
    Dma(u8),
    TileCtrl,
    Fifo(u8),
    South(u8),
    West(u8),
    North(u8),
    East(u8),
    Trace(u8),
}

/// A group of contiguous physical port indices. Zero count means absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortGroup {
    pub first: u8,
    pub count: u8,
}

// Order: Core, DMA, TileCtrl, FIFO, South, West, North, East, Trace.
const fn g(first: u8, count: u8) -> PortGroup {
    PortGroup { first, count }
}
pub const COMPUTE_MASTERS: [PortGroup; 9] = [
    g(0, 1),
    g(1, 2),
    g(3, 1),
    g(4, 1),
    g(5, 4),
    g(9, 4),
    g(13, 6),
    g(19, 4),
    g(0, 0),
];
pub const COMPUTE_SLAVES: [PortGroup; 9] = [
    g(0, 1),
    g(1, 2),
    g(3, 1),
    g(4, 1),
    g(5, 6),
    g(11, 4),
    g(15, 4),
    g(19, 4),
    g(23, 2),
];
pub const MEMTILE_MASTERS: [PortGroup; 9] = [
    g(0, 0),
    g(0, 6),
    g(6, 1),
    g(0, 0),
    g(7, 4),
    g(0, 0),
    g(11, 6),
    g(0, 0),
    g(0, 0),
];
pub const MEMTILE_SLAVES: [PortGroup; 9] = [
    g(0, 0),
    g(0, 6),
    g(6, 1),
    g(0, 0),
    g(7, 6),
    g(0, 0),
    g(13, 4),
    g(0, 0),
    g(17, 1),
];
pub const SHIM_MASTERS: [PortGroup; 9] = [
    g(0, 0),
    g(0, 0),
    g(0, 1),
    g(1, 1),
    g(2, 6),
    g(8, 4),
    g(12, 6),
    g(18, 4),
    g(0, 0),
];
pub const SHIM_SLAVES: [PortGroup; 9] = [
    g(0, 0),
    g(0, 0),
    g(0, 1),
    g(1, 1),
    g(2, 8),
    g(10, 4),
    g(14, 4),
    g(18, 4),
    g(22, 1),
];

pub fn port_index(kind: TileType, port: Port, master: bool) -> u32 {
    let (group, number) = match port {
        Port::Core(n) => (0, n),
        Port::Dma(n) => (1, n),
        Port::TileCtrl => (2, 0),
        Port::Fifo(n) => (3, n),
        Port::South(n) => (4, n),
        Port::West(n) => (5, n),
        Port::North(n) => (6, n),
        Port::East(n) => (7, n),
        Port::Trace(n) => (8, n),
    };
    let table = match (kind, master) {
        (TileType::Compute, true) => &COMPUTE_MASTERS,
        (TileType::Compute, false) => &COMPUTE_SLAVES,
        (TileType::Memtile, true) => &MEMTILE_MASTERS,
        (TileType::Memtile, false) => &MEMTILE_SLAVES,
        (TileType::Shim, true) => &SHIM_MASTERS,
        (TileType::Shim, false) => &SHIM_SLAVES,
    };
    let entry = table[group];
    assert!(
        number < entry.count,
        "port absent/out of range: {kind:?} {port:?}, master={master}"
    );
    (entry.first + number) as u32
}

fn switch_base(tile: Location) -> u32 {
    if tile.kind() == TileType::Memtile {
        0xb0000
    } else {
        0x3f000
    }
}
fn port_addr(tile: Location, port: Port, master: bool) -> u32 {
    tile.address(
        switch_base(tile)
            + if master { 0 } else { 0x100 }
            + 4 * port_index(tile.kind(), port, master),
    )
}

#[derive(Clone, Copy, Debug)]
pub struct Circuit {
    pub tile: Location,
    pub slave: Port,
    pub master: Port,
}
impl Circuit {
    /// Master then slave, matching aie-rt; repeated slave writes preserve multicast order.
    pub fn writes(self) -> [(u32, u32); 2] {
        [
            (
                port_addr(self.tile, self.master, true),
                1 << 31 | port_index(self.tile.kind(), self.slave, false),
            ),
            (port_addr(self.tile, self.slave, false), 1 << 31),
        ]
    }
    pub fn emit_cdo(self, cdo: &mut Cdo) {
        for (addr, value) in self.writes() {
            cdo.write(addr, value);
        }
    }
    pub fn emit_txn(self, txn: &mut Txn) {
        for (addr, value) in self.writes() {
            txn.write32(addr, value);
        }
    }
}

/// One packet-master output can accept any enabled master-select bit for an arbiter.
#[derive(Clone, Copy, Debug)]
pub struct PacketMaster {
    pub tile: Location,
    pub port: Port,
    pub arbiter: u8,
    pub selects: u8,
    pub drop_header: bool,
}
impl PacketMaster {
    pub fn write(self) -> (u32, u32) {
        assert!(self.arbiter < 8 && self.selects < 16);
        (
            port_addr(self.tile, self.port, true),
            3 << 30
                | (self.drop_header as u32) << 7
                | (self.selects as u32) << 3
                | self.arbiter as u32,
        )
    }
    pub fn emit_cdo(self, cdo: &mut Cdo) {
        let (a, v) = self.write();
        cdo.write(a, v);
    }
    pub fn emit_txn(self, txn: &mut Txn) {
        let (a, v) = self.write();
        txn.write32(a, v);
    }
}

/// Packet ID matching (type is not part of a hardware stream-switch rule).
#[derive(Clone, Copy, Debug)]
pub struct PacketRule {
    pub tile: Location,
    pub slave: Port,
    pub slot: u8,
    pub id: u8,
    pub mask: u8,
    pub select: u8,
    pub arbiter: u8,
}
impl PacketRule {
    pub fn writes(self) -> [(u32, u32); 2] {
        assert!(
            self.slot < 4 && self.id < 32 && self.mask < 32 && self.select < 4 && self.arbiter < 8
        );
        let index = port_index(self.tile.kind(), self.slave, false);
        [
            (port_addr(self.tile, self.slave, false), 3 << 30),
            (
                self.tile
                    .address(switch_base(self.tile) + 0x200 + index * 16 + self.slot as u32 * 4),
                (self.id as u32) << 24
                    | (self.mask as u32) << 16
                    | 1 << 8
                    | (self.select as u32) << 4
                    | self.arbiter as u32,
            ),
        ]
    }
    pub fn emit_cdo(self, cdo: &mut Cdo) {
        for (a, v) in self.writes() {
            cdo.write(a, v);
        }
    }
    pub fn emit_txn(self, txn: &mut Txn) {
        for (a, v) in self.writes() {
            txn.write32(a, v);
        }
    }
}

/// Connect one shim DMA channel to the stream switch's south-facing port.
#[derive(Clone, Copy, Debug)]
pub struct ShimDma {
    pub tile: Location,
    pub direction: crate::dma::Direction,
    pub channel: u8,
}
impl ShimDma {
    pub fn port(self) -> Port {
        assert!(self.tile.kind() == TileType::Shim && self.channel < 2);
        match (self.direction, self.channel) {
            (crate::dma::Direction::Mm2s, 0) => Port::South(3),
            (crate::dma::Direction::Mm2s, _) => Port::South(7),
            (crate::dma::Direction::S2mm, 0) => Port::South(2),
            (crate::dma::Direction::S2mm, _) => Port::South(3),
        }
    }
    /// `(address, mask, value)`, selecting the DMA (selection 1) in the NOC mux.
    pub fn write(self) -> (u32, u32, u32) {
        self.port();
        let (offset, shift) = match (self.direction, self.channel) {
            (crate::dma::Direction::Mm2s, 0) => (0x1f000, 10),
            (crate::dma::Direction::Mm2s, _) => (0x1f000, 14),
            (crate::dma::Direction::S2mm, 0) => (0x1f004, 4),
            (crate::dma::Direction::S2mm, _) => (0x1f004, 6),
        };
        (self.tile.address(offset), 3 << shift, 1 << shift)
    }
    pub fn emit_cdo(self, cdo: &mut Cdo) {
        let (a, m, v) = self.write();
        cdo.mask_write(a, m, v);
    }
    pub fn emit_txn(self, txn: &mut Txn) {
        let (a, m, v) = self.write();
        txn.mask_write(a, v, m);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dma::vendor_gate::{self, CORPUS};
    use Port::*;

    // Physical circuit graph of whole_array (8 columns x 4 cores), transcribed
    // from the placed vendor routing configuration. Each pair is (slave,master),
    // not a register write. Do not derive the input graph from the oracle at runtime.
    fn whole_array_routes(cdo: &mut Cdo, txn: &mut Txn) {
        let graph: &[(u32, u32, &[(Port, Port)])] = &[
            (0, 0, &[(South(3), North(5)), (North(2), South(2))]),
            (
                0,
                1,
                &[
                    (Dma(0), North(1)),
                    (South(5), Dma(0)),
                    (North(3), Dma(1)),
                    (North(1), Dma(2)),
                    (North(0), Dma(3)),
                    (North(2), Dma(4)),
                    (Dma(1), South(2)),
                ],
            ),
            (
                0,
                2,
                &[
                    (South(1), Dma(0)),
                    (South(1), North(1)),
                    (South(1), East(0)),
                    (East(0), Dma(1)),
                    (Dma(0), East(2)),
                    (North(2), South(3)),
                    (North(3), South(1)),
                    (North(0), South(0)),
                    (East(1), South(2)),
                ],
            ),
            (
                0,
                3,
                &[
                    (South(1), Dma(0)),
                    (South(1), North(5)),
                    (East(0), Dma(1)),
                    (Dma(0), East(2)),
                    (North(0), South(2)),
                    (North(2), South(3)),
                    (North(1), South(0)),
                ],
            ),
            (
                0,
                4,
                &[
                    (South(5), Dma(0)),
                    (South(5), North(5)),
                    (South(5), East(2)),
                    (East(2), Dma(1)),
                    (Dma(0), East(1)),
                    (North(2), South(0)),
                    (North(1), South(2)),
                    (East(1), South(1)),
                ],
            ),
            (
                0,
                5,
                &[
                    (South(5), Dma(0)),
                    (East(0), Dma(1)),
                    (Dma(0), East(1)),
                    (East(1), South(2)),
                    (East(3), South(1)),
                ],
            ),
            (1, 0, &[(South(3), North(1)), (North(2), South(2))]),
            (
                1,
                1,
                &[
                    (Dma(0), North(1)),
                    (South(1), Dma(0)),
                    (North(0), Dma(1)),
                    (North(2), Dma(2)),
                    (North(1), Dma(3)),
                    (North(3), Dma(4)),
                    (Dma(1), South(2)),
                ],
            ),
            (
                1,
                2,
                &[
                    (West(0), Dma(0)),
                    (West(0), North(0)),
                    (East(0), West(0)),
                    (East(1), Dma(1)),
                    (South(1), North(4)),
                    (West(2), East(2)),
                    (North(3), South(0)),
                    (North(0), South(2)),
                    (North(1), South(1)),
                    (East(3), South(3)),
                    (Dma(0), East(3)),
                    (East(2), West(1)),
                ],
            ),
            (
                1,
                3,
                &[
                    (South(0), Dma(0)),
                    (East(1), West(0)),
                    (East(0), Dma(1)),
                    (South(4), North(2)),
                    (West(2), East(1)),
                    (North(1), South(3)),
                    (North(3), South(0)),
                    (North(2), South(1)),
                    (Dma(0), East(2)),
                ],
            ),
            (
                1,
                4,
                &[
                    (West(2), Dma(0)),
                    (West(2), North(2)),
                    (East(0), West(2)),
                    (East(2), Dma(1)),
                    (South(2), North(4)),
                    (West(1), East(1)),
                    (North(3), South(1)),
                    (North(2), South(3)),
                    (North(1), South(2)),
                    (Dma(0), East(0)),
                    (East(3), West(1)),
                ],
            ),
            (
                1,
                5,
                &[
                    (South(2), Dma(0)),
                    (East(0), West(0)),
                    (South(4), Dma(1)),
                    (South(4), East(2)),
                    (West(1), South(3)),
                    (East(1), South(2)),
                    (East(3), South(1)),
                    (Dma(0), West(1)),
                    (East(2), West(3)),
                ],
            ),
            (
                2,
                0,
                &[
                    (South(3), North(1)),
                    (South(7), North(5)),
                    (North(2), South(2)),
                ],
            ),
            (
                2,
                1,
                &[
                    (Dma(0), North(1)),
                    (South(1), Dma(0)),
                    (Dma(1), North(5)),
                    (South(5), Dma(1)),
                    (North(3), Dma(2)),
                    (North(1), Dma(3)),
                    (North(0), Dma(4)),
                    (North(2), Dma(5)),
                    (Dma(2), South(2)),
                ],
            ),
            (
                2,
                2,
                &[
                    (South(1), Dma(0)),
                    (South(1), North(4)),
                    (South(1), East(0)),
                    (East(0), West(0)),
                    (East(0), Dma(1)),
                    (South(5), North(5)),
                    (East(3), West(1)),
                    (West(2), East(3)),
                    (Dma(0), East(1)),
                    (North(1), South(3)),
                    (North(2), South(1)),
                    (East(2), South(0)),
                    (East(1), South(2)),
                    (North(0), West(3)),
                    (West(3), East(2)),
                    (North(3), West(2)),
                ],
            ),
            (
                2,
                3,
                &[
                    (South(4), Dma(0)),
                    (South(4), North(3)),
                    (South(4), East(2)),
                    (East(1), West(1)),
                    (East(1), Dma(1)),
                    (South(5), North(5)),
                    (East(3), West(0)),
                    (West(1), South(1)),
                    (Dma(0), South(2)),
                    (North(1), East(0)),
                    (North(0), South(0)),
                    (West(2), East(1)),
                    (North(2), South(3)),
                ],
            ),
            (
                2,
                4,
                &[
                    (South(3), Dma(0)),
                    (South(3), North(0)),
                    (South(5), West(0)),
                    (South(5), Dma(1)),
                    (South(5), East(1)),
                    (East(2), West(2)),
                    (West(1), South(1)),
                    (Dma(0), East(2)),
                    (East(3), South(0)),
                    (West(0), East(3)),
                    (East(1), West(3)),
                    (East(0), South(2)),
                ],
            ),
            (
                2,
                5,
                &[
                    (South(0), Dma(0)),
                    (South(0), East(3)),
                    (East(1), West(0)),
                    (East(1), Dma(1)),
                    (West(2), East(0)),
                    (Dma(0), West(1)),
                    (East(0), West(3)),
                    (East(2), West(2)),
                ],
            ),
            (
                3,
                0,
                &[
                    (South(3), North(1)),
                    (South(7), North(5)),
                    (North(2), South(2)),
                ],
            ),
            (
                3,
                1,
                &[
                    (Dma(0), North(1)),
                    (South(1), Dma(0)),
                    (Dma(1), North(5)),
                    (South(5), Dma(1)),
                    (North(3), Dma(2)),
                    (North(1), Dma(3)),
                    (North(2), Dma(4)),
                    (North(0), Dma(5)),
                    (Dma(2), South(2)),
                ],
            ),
            (
                3,
                2,
                &[
                    (West(0), Dma(0)),
                    (South(1), West(0)),
                    (South(1), East(1)),
                    (South(5), North(0)),
                    (East(2), West(3)),
                    (East(2), Dma(1)),
                    (West(3), South(3)),
                    (West(1), South(1)),
                    (East(3), South(2)),
                    (East(1), South(0)),
                    (North(2), West(2)),
                    (North(0), West(1)),
                    (North(3), East(0)),
                    (West(2), East(3)),
                    (Dma(0), East(2)),
                ],
            ),
            (
                3,
                3,
                &[
                    (West(2), Dma(0)),
                    (West(2), North(2)),
                    (South(0), West(1)),
                    (South(0), East(2)),
                    (East(0), West(3)),
                    (East(0), Dma(1)),
                    (East(3), South(2)),
                    (East(1), South(0)),
                    (West(0), South(3)),
                    (North(1), East(1)),
                    (West(1), East(0)),
                    (Dma(0), East(3)),
                ],
            ),
            (
                3,
                4,
                &[
                    (South(2), Dma(0)),
                    (West(1), East(0)),
                    (East(1), North(5)),
                    (East(3), West(2)),
                    (East(3), Dma(1)),
                    (West(2), South(1)),
                    (North(3), West(3)),
                    (West(3), East(2)),
                    (Dma(0), East(3)),
                    (North(0), West(1)),
                    (North(2), West(0)),
                ],
            ),
            (
                3,
                5,
                &[
                    (West(3), Dma(0)),
                    (South(5), West(1)),
                    (West(0), Dma(1)),
                    (West(0), East(0)),
                    (East(3), West(0)),
                    (East(0), South(3)),
                    (Dma(0), West(2)),
                    (East(1), South(0)),
                    (East(2), South(2)),
                ],
            ),
            (
                4,
                0,
                &[
                    (South(3), North(1)),
                    (South(7), North(5)),
                    (North(2), South(2)),
                ],
            ),
            (
                4,
                1,
                &[
                    (Dma(0), North(1)),
                    (South(1), Dma(0)),
                    (Dma(1), North(5)),
                    (South(5), Dma(1)),
                    (North(0), Dma(2)),
                    (North(3), Dma(3)),
                    (North(2), Dma(4)),
                    (North(1), Dma(5)),
                    (Dma(2), South(2)),
                ],
            ),
            (
                4,
                2,
                &[
                    (South(1), Dma(0)),
                    (South(1), North(4)),
                    (South(1), East(3)),
                    (West(1), Dma(1)),
                    (West(1), East(0)),
                    (South(5), North(0)),
                    (East(3), West(2)),
                    (Dma(0), West(3)),
                    (East(0), West(1)),
                    (West(0), South(0)),
                    (North(1), South(3)),
                    (North(0), South(2)),
                    (East(2), South(1)),
                    (West(3), East(1)),
                    (West(2), East(2)),
                ],
            ),
            (
                4,
                3,
                &[
                    (South(4), Dma(0)),
                    (South(4), North(4)),
                    (South(4), East(1)),
                    (West(2), Dma(1)),
                    (West(2), East(3)),
                    (South(0), North(1)),
                    (East(0), West(0)),
                    (Dma(0), West(3)),
                    (East(1), West(1)),
                    (West(1), South(1)),
                    (North(1), South(0)),
                    (West(0), East(2)),
                    (West(3), East(0)),
                ],
            ),
            (
                4,
                4,
                &[
                    (South(4), Dma(0)),
                    (South(4), North(2)),
                    (South(4), East(0)),
                    (West(0), Dma(1)),
                    (West(0), East(1)),
                    (South(1), West(1)),
                    (South(1), North(4)),
                    (East(3), West(3)),
                    (Dma(0), South(1)),
                    (West(2), East(3)),
                    (West(3), East(2)),
                ],
            ),
            (
                4,
                5,
                &[
                    (South(2), Dma(0)),
                    (South(2), East(3)),
                    (South(4), Dma(1)),
                    (South(4), East(1)),
                    (West(0), East(0)),
                    (Dma(0), West(3)),
                    (East(1), West(0)),
                    (East(0), West(1)),
                    (East(3), West(2)),
                ],
            ),
            (
                5,
                0,
                &[
                    (South(3), North(1)),
                    (South(7), North(5)),
                    (North(2), South(2)),
                ],
            ),
            (
                5,
                1,
                &[
                    (Dma(0), North(1)),
                    (South(1), Dma(0)),
                    (Dma(1), North(5)),
                    (South(5), Dma(1)),
                    (North(3), Dma(2)),
                    (North(0), Dma(3)),
                    (North(2), Dma(4)),
                    (North(1), Dma(5)),
                    (Dma(2), South(2)),
                ],
            ),
            (
                5,
                2,
                &[
                    (West(3), Dma(0)),
                    (West(0), East(2)),
                    (South(1), West(3)),
                    (South(1), Dma(1)),
                    (South(1), East(1)),
                    (South(5), North(5)),
                    (East(3), West(0)),
                    (North(2), West(2)),
                    (West(1), South(3)),
                    (West(2), South(0)),
                    (Dma(0), South(2)),
                    (East(1), South(1)),
                    (North(3), East(0)),
                    (North(1), East(3)),
                ],
            ),
            (
                5,
                3,
                &[
                    (West(1), Dma(0)),
                    (West(3), East(0)),
                    (South(5), West(0)),
                    (South(5), Dma(1)),
                    (South(5), East(3)),
                    (East(1), North(5)),
                    (East(2), West(1)),
                    (North(0), South(2)),
                    (West(2), South(3)),
                    (West(0), East(2)),
                    (Dma(0), East(1)),
                    (North(1), South(1)),
                ],
            ),
            (
                5,
                4,
                &[
                    (West(0), Dma(0)),
                    (West(1), East(1)),
                    (South(5), West(3)),
                    (South(5), Dma(1)),
                    (East(3), South(0)),
                    (West(3), South(1)),
                    (West(2), East(0)),
                    (Dma(0), East(3)),
                ],
            ),
            (
                5,
                5,
                &[
                    (West(3), Dma(0)),
                    (West(1), East(3)),
                    (West(0), Dma(1)),
                    (West(0), East(2)),
                    (East(1), West(1)),
                    (Dma(0), West(0)),
                    (East(0), West(3)),
                ],
            ),
            (
                6,
                0,
                &[
                    (South(3), North(1)),
                    (South(7), North(5)),
                    (North(2), South(2)),
                ],
            ),
            (
                6,
                1,
                &[
                    (Dma(0), North(1)),
                    (South(1), Dma(0)),
                    (Dma(1), North(5)),
                    (South(5), Dma(1)),
                    (North(2), Dma(2)),
                    (North(3), Dma(3)),
                    (North(0), Dma(4)),
                    (North(1), Dma(5)),
                    (Dma(2), South(2)),
                ],
            ),
            (
                6,
                2,
                &[
                    (South(1), Dma(0)),
                    (South(1), North(0)),
                    (South(1), East(1)),
                    (West(2), Dma(1)),
                    (West(1), East(0)),
                    (South(5), North(5)),
                    (Dma(0), West(3)),
                    (East(3), West(1)),
                    (West(0), South(2)),
                    (North(1), South(3)),
                    (North(3), South(0)),
                    (North(2), South(1)),
                    (West(3), East(3)),
                    (North(0), East(2)),
                ],
            ),
            (
                6,
                3,
                &[
                    (South(0), Dma(0)),
                    (South(0), North(2)),
                    (West(0), Dma(1)),
                    (West(3), East(3)),
                    (South(5), West(1)),
                    (South(5), North(1)),
                    (Dma(0), West(2)),
                    (West(2), South(1)),
                    (West(1), South(3)),
                    (East(3), South(2)),
                    (North(1), East(2)),
                    (North(0), South(0)),
                ],
            ),
            (
                6,
                4,
                &[
                    (South(2), Dma(0)),
                    (South(2), North(0)),
                    (South(2), East(1)),
                    (West(1), Dma(1)),
                    (South(1), East(2)),
                    (Dma(0), West(3)),
                    (West(0), South(1)),
                    (West(3), South(0)),
                ],
            ),
            (
                6,
                5,
                &[
                    (South(0), Dma(0)),
                    (South(0), East(3)),
                    (West(3), Dma(1)),
                    (West(2), East(2)),
                    (Dma(0), West(1)),
                    (East(3), West(0)),
                ],
            ),
            (7, 0, &[(North(2), South(2))]),
            (
                7,
                1,
                &[
                    (North(2), Dma(0)),
                    (North(1), Dma(1)),
                    (North(3), Dma(2)),
                    (North(0), Dma(3)),
                    (Dma(0), South(2)),
                ],
            ),
            (
                7,
                2,
                &[
                    (West(1), Dma(0)),
                    (West(1), North(4)),
                    (West(0), Dma(1)),
                    (Dma(0), West(3)),
                    (West(3), South(2)),
                    (North(0), South(1)),
                    (West(2), South(3)),
                    (North(1), South(0)),
                ],
            ),
            (
                7,
                3,
                &[
                    (South(4), Dma(0)),
                    (West(3), Dma(1)),
                    (Dma(0), West(3)),
                    (West(2), South(0)),
                    (North(3), South(1)),
                ],
            ),
            (
                7,
                4,
                &[(West(1), Dma(0)), (West(2), Dma(1)), (Dma(0), South(3))],
            ),
            (
                7,
                5,
                &[(West(3), Dma(0)), (West(2), Dma(1)), (Dma(0), West(3))],
            ),
        ];
        for &(col, row, pairs) in graph {
            let tile = Location::new(col, row);
            for &(slave, master) in pairs {
                let circuit = Circuit {
                    tile,
                    slave,
                    master,
                };
                circuit.emit_cdo(cdo);
                circuit.emit_txn(txn);
            }
        }
        // Shim controller task-complete packet ID 15 -> south0, arbiter5/msel3.
        for col in 0..8 {
            let tile = Location::new(col, 0);
            let master = PacketMaster {
                tile,
                port: South(0),
                arbiter: 5,
                selects: 8,
                drop_header: false,
            };
            let rule = PacketRule {
                tile,
                slave: TileCtrl,
                slot: 0,
                id: 15,
                mask: 31,
                select: 3,
                arbiter: 5,
            };
            master.emit_cdo(cdo);
            master.emit_txn(txn);
            rule.emit_cdo(cdo);
            rule.emit_txn(txn);
        }
    }

    fn switch_register(address: u32) -> bool {
        let off = address & 0xfffff;
        (0x3f000..0x3f400).contains(&off) || (0xb0000..0xb0400).contains(&off)
    }

    #[test]
    fn reproduces_vendor_whole_array_routes() {
        let Some(corpus) = crate::vendor_corpus() else {
            return;
        };
        let root = corpus.join("cache").join(CORPUS);
        let root = root.as_path();
        let original = std::fs::read(root.join("cdo_main/main_aie_cdo_init.bin")).unwrap();
        let expected: Vec<_> = vendor_gate::cdo_writes(&vendor_gate::words(&original))
            .into_iter()
            .filter(|&(a, _)| switch_register(a))
            .collect();
        let mut cdo = Cdo::new();
        let mut txn = Txn::aie2p_8col();
        whole_array_routes(&mut cdo, &mut txn);
        vendor_gate::compare(
            vendor_gate::cdo_writes(&cdo.to_words()),
            expected.clone(),
            "CDO stream switch",
        );
        vendor_gate::compare(
            vendor_gate::txn_writes(&txn.to_bytes()),
            expected,
            "TXN stream switch",
        );
        // Runtime insts contain no stream-switch writes; they inherit CDO routes.
        let original = std::fs::read(root.join("insts.bin")).unwrap();
        assert!(vendor_gate::txn_writes(&original)
            .iter()
            .all(|&(a, _)| !switch_register(a)));
    }
}
