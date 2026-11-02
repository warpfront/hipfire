// SPDX-License-Identifier: Apache-2.0 AND MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// Register offsets transcribed from AMD/Xilinx aie-rt (MIT); see NOTICE.
//! AIE2P tile addressing and register offsets (aie-rt `release/main_aig`,
//! `driver/src/global/xaie2pgbl_params.h`, prefix `XAIE2PGBL_`).
//!
//! Array address = col << 25 | row << 20 | offset. Row 0 = shim (NOC/PL modules), row 1 = mem tile,
//! rows 2.. = compute tiles (core + memory module). Columns are partition-relative.

pub const COL_SHIFT: u32 = 25;
pub const ROW_SHIFT: u32 = 20;

pub const fn tile(col: u32, row: u32, off: u32) -> u32 {
    (col << COL_SHIFT) | (row << ROW_SHIFT) | off
}

/// Compute tile (rows >= 2).
pub mod core {
    /// CORE_MODULE_PROGRAM_MEMORY (16 KiB)
    pub const PROGRAM_MEMORY: u32 = 0x0002_0000;
    /// CORE_MODULE_CORE_CONTROL: bit0 ENABLE, bit1 RESET
    pub const CORE_CONTROL: u32 = 0x0003_2000;
    /// CORE_MODULE_CORE_STATUS: bit20 CORE_DONE, bit19 ERROR_HALT, bits6..9 lock stalls
    pub const CORE_STATUS: u32 = 0x0003_2004;
    /// CORE_MODULE_CORE_PC
    pub const CORE_PC: u32 = 0x0003_0E00;
    /// MEMORY_MODULE_LOCKi_VALUE = LOCK0_VALUE + 0x10 * i (16 locks)
    pub const LOCK0_VALUE: u32 = 0x0001_F000;
    /// MEMORY_MODULE_DMA_BDi_0 = BD0 + 0x20 * i (16 BDs x 6 words)
    pub const DMA_BD0: u32 = 0x0001_D000;
    /// MEMORY_MODULE_DMA_{S2MM_0,S2MM_1,MM2S_0,MM2S_1}_CTRL (bit1 RESET)
    pub const DMA_S2MM_0_CTRL: u32 = 0x0001_DE00;
    pub const DMA_S2MM_1_CTRL: u32 = 0x0001_DE08;
    pub const DMA_MM2S_0_CTRL: u32 = 0x0001_DE10;
    pub const DMA_MM2S_1_CTRL: u32 = 0x0001_DE18;
    /// ..._START_QUEUE: bit31 ENABLE_TOKEN_ISSUE, bits16..23 REPEAT_COUNT, bits0..3 START_BD_ID
    pub const DMA_S2MM_0_START_QUEUE: u32 = 0x0001_DE04;
    pub const DMA_MM2S_0_START_QUEUE: u32 = 0x0001_DE14;
    /// CORE_MODULE_STREAM_SWITCH_MASTER_CONFIG_*: bit31 enable, bits0..6 slave index
    pub const SS_MASTER_DMA0: u32 = 0x0003_F004;
    pub const SS_MASTER_SOUTH0: u32 = 0x0003_F014;
    /// CORE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_AIE_CORE0 is slave index 0
    pub const SS_SLAVE_BASE: u32 = 0x0003_F100;
    pub const SS_SLAVE_DMA_0: u32 = 0x0003_F104;
    pub const SS_SLAVE_SOUTH_1: u32 = 0x0003_F118;
    /// Core-view base of the tile's own data memory (lock ids 48..63 address its locks).
    pub const CORE_VIEW_OWN_MEM: u32 = 0x0007_0000;
    pub const CORE_LOCK_OWN_BASE: u64 = 48;
}

/// Memory tile (row 1).
pub mod mem {
    /// MEM_TILE_MODULE_STREAM_SWITCH_MASTER_CONFIG_SOUTH0 / NORTH1
    pub const SS_MASTER_SOUTH0: u32 = 0x000B_001C;
    pub const SS_MASTER_NORTH1: u32 = 0x000B_0030;
    /// MEM_TILE_MODULE_STREAM_SWITCH_SLAVE_CONFIG_DMA_0 is slave index 0
    pub const SS_SLAVE_BASE: u32 = 0x000B_0100;
    pub const SS_SLAVE_SOUTH_1: u32 = 0x000B_0120;
    pub const SS_SLAVE_NORTH_0: u32 = 0x000B_0134;
}

/// Shim NOC tile (row 0).
pub mod shim {
    /// NOC_MODULE_DMA_BDi_0 = BD0 + 0x20 * i (16 BDs x 8 words)
    pub const DMA_BD0: u32 = 0x0001_D000;
    /// NOC_MODULE_DMA_S2MM_0_CTRL: bits8..15 CONTROLLER_ID (token routing)
    pub const DMA_S2MM_0_CTRL: u32 = 0x0001_D200;
    /// NOC_MODULE_DMA_S2MM_0_TASK_QUEUE: bit31 ENABLE_TOKEN_ISSUE, bits0..3 START_BD_ID
    pub const DMA_S2MM_0_TASK_QUEUE: u32 = 0x0001_D204;
    pub const DMA_MM2S_0_CTRL: u32 = 0x0001_D210;
    pub const DMA_MM2S_0_TASK_QUEUE: u32 = 0x0001_D214;
    /// NOC_MODULE_MUX_CONFIG / DEMUX_CONFIG
    pub const MUX_CONFIG: u32 = 0x0001_F000;
    pub const DEMUX_CONFIG: u32 = 0x0001_F004;
    /// NOC_MODULE_LOCKi_VALUE
    pub const LOCK0_VALUE: u32 = 0x0001_4000;
    /// PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_SOUTH2 / NORTH1
    pub const SS_MASTER_SOUTH2: u32 = 0x0003_F010;
    pub const SS_MASTER_NORTH1: u32 = 0x0003_F034;
    /// PL_MODULE_STREAM_SWITCH_SLAVE_CONFIG_TILE_CTRL is slave index 0
    pub const SS_SLAVE_BASE: u32 = 0x0003_F100;
    pub const SS_SLAVE_SOUTH_3: u32 = 0x0003_F114;
    pub const SS_SLAVE_NORTH_0: u32 = 0x0003_F138;
    /// PL_MODULE_STREAM_SWITCH_MASTER_CONFIG_SOUTH0: carries tile-control packets (DMA
    /// task-complete tokens, controller id 0xF) south to the array controller.
    pub const SS_MASTER_SOUTH0: u32 = 0x0003_F008;
    /// PL_MODULE_STREAM_SWITCH_SLAVE_CONFIG_TILE_CTRL (slave index 0) and its packet slot 0.
    pub const SS_SLAVE_TILE_CTRL: u32 = 0x0003_F100;
    pub const SS_SLAVE_TILE_CTRL_SLOT0: u32 = 0x0003_F200;
}

pub const SS_ENABLE: u32 = 1 << 31;

/// Stream-switch master config value routing `slave_reg` (circuit-switched).
pub const fn ss_master(slave_reg: u32, slave_base: u32) -> u32 {
    SS_ENABLE | ((slave_reg - slave_base) / 4)
}

pub const SS_PACKET: u32 = 1 << 30;

/// Packet-switched master config: arbiter `arbit` (0..7) fed by msel `msel` (0..3).
/// CONFIGURATION[6:3] = msel enable mask, [2:0] = arbiter.
pub const fn ss_master_packet(msel: u32, arbit: u32) -> u32 {
    SS_ENABLE | SS_PACKET | (1 << (3 + msel)) | arbit
}

/// Slave packet slot: match packet `id` under `mask`, forward to (msel, arbiter).
pub const fn ss_slot(id: u32, mask: u32, msel: u32, arbit: u32) -> u32 {
    (id << 24) | (mask << 16) | (1 << 8) | (msel << 4) | arbit
}

/// Writes routing a tile's TILE_CTRL packets (DMA task-complete tokens; controller id 0xF in the
/// DMA channel CTRL register) out of the shim's SOUTH0 to the array controller. Without it the
/// data moves but TXN `sync` never sees the token.
pub fn shim_token_route(col: u32) -> [(u32, u32); 3] {
    [
        (tile(col, 0, shim::SS_MASTER_SOUTH0), ss_master_packet(3, 5)),
        (
            tile(col, 0, shim::SS_SLAVE_TILE_CTRL),
            SS_ENABLE | SS_PACKET,
        ),
        (
            tile(col, 0, shim::SS_SLAVE_TILE_CTRL_SLOT0),
            ss_slot(0xF, 0x1F, 3, 5),
        ),
    ]
}

#[cfg(test)]
mod tests {
    /// Values from the vendor add_one PDI (mlir-aie npu2).
    #[test]
    fn token_route_matches_vendor() {
        assert_eq!(
            super::shim_token_route(0),
            [
                (0x0003_F008, 0xC000_0045),
                (0x0003_F100, 0xC000_0000),
                (0x0003_F200, 0x0F1F_0135)
            ]
        );
    }
}
