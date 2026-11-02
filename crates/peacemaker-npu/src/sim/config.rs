// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Execute submitted CDO/TXN configuration, not a hand-wired replacement.
//! Circuit streams are routed from actual enabled master/slave registers across
//! cardinal tile links and shim NOC muxes. TILE_CTRL task-complete tokens
//! require a matching packet rule to SOUTH0, as on hardware. Compute/memtile/shim
//! tensor BDs are supported; other packet streams are rejected.
//! DMA finish-on-TLAST (CTRL bits16..17) must stay disabled: descriptors are
//! length-driven and FIFOs carry payload words, not packet framing. Own memtile
//! memory/locks are supported; neighboring DMA views and padding are rejected.
//! Channel control permits only compute ENABLE/RESET, memtile RESET and the
//! controller ID. Shim pause/FoT/order/compression and non-DMA NOC selections
//! fail explicitly rather than pretending to execute another mode.
use crate::sim::{
    decode::PreparedDecoder,
    dma::{self, Access, Descriptor, Direction, Fifo},
    Neighbour, Neighbours, Step, Tile,
};
use crate::{
    cdo::{self, Cmd},
    dma::{Location, TileType},
    pdi, regs,
    route::{self, Port},
};
use std::collections::{BTreeMap, VecDeque};
pub type Result<T, E = String> = std::result::Result<T, E>;
fn error(e: crate::sim::Error) -> String {
    format!("{e:?}")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TxnOp {
    Write(u32, u32),
    BlockWrite(u32, Vec<u32>),
    MaskWrite(u32, u32, u32),
    MaskPoll(u32, u32, u32),
    DdrPatch {
        address: u32,
        arg: usize,
        plus: u64,
    },
    Sync {
        col: u8,
        row: u8,
        direction: u8,
        channel: u8,
        cols: u8,
        rows: u8,
    },
}
/// Checked parser for the firmware v0.1 transaction operations emitted by peacemaker-npu.
pub fn parse_txn(bytes: &[u8]) -> Result<Vec<TxnOp>> {
    if bytes.len() < 16
        || bytes.len() % 4 != 0
        || bytes[..3] != [0, 1, 4]
        || bytes[3..6] != [6, 8, 1]
    {
        return Err("unsupported TXN header".into());
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if words[3] as usize != bytes.len() {
        return Err("TXN size mismatch".into());
    }
    if words[2] as usize > words.len() - 4 {
        return Err("invalid TXN operation count".into());
    }
    let mut index = 4;
    let mut ops = Vec::with_capacity(words[2] as usize);
    for _ in 0..words[2] {
        let id = *words.get(index).ok_or("truncated TXN operation")?;
        let length = match id {
            0 => 6,
            1 => {
                let n = *words.get(index + 3).ok_or("truncated block write")?;
                if n < 16 || n % 4 != 0 {
                    return Err("invalid block write size".into());
                }
                n as usize / 4
            }
            3 | 4 => 7,
            0x80 => 4,
            0x81 => 12,
            _ => return Err(format!("unsupported TXN opcode {id:#x}")),
        };
        let p = words
            .get(index..index.checked_add(length).ok_or("TXN length overflow")?)
            .ok_or("truncated TXN payload")?;
        let op = match id {
            0 if p[5] == 24 && p[3] == 0 => TxnOp::Write(p[2], p[4]),
            1 => TxnOp::BlockWrite(p[2], p[4..].to_vec()),
            3 | 4 if p[6] == 28 && p[3] == 0 => {
                if id == 3 {
                    TxnOp::MaskWrite(p[2], p[5], p[4])
                } else {
                    TxnOp::MaskPoll(p[2], p[5], p[4])
                }
            }
            0x80 if p[1] == 16 => TxnOp::Sync {
                col: (p[2] >> 16) as u8,
                row: (p[2] >> 8) as u8,
                direction: p[2] as u8,
                channel: (p[3] >> 24) as u8,
                cols: (p[3] >> 16) as u8,
                rows: (p[3] >> 8) as u8,
            },
            0x81 if p[1] == 48 && p[7] == 0 && p[9] == 0 => TxnOp::DdrPatch {
                address: p[6],
                arg: p[8] as usize,
                plus: p[10] as u64 | ((p[11] as u64) << 32),
            },
            _ => return Err(format!("malformed TXN opcode {id:#x}")),
        };
        ops.push(op);
        index += length;
    }
    if index != words.len() {
        return Err("trailing TXN operations".into());
    }
    Ok(ops)
}
// Guard peacemaker-npu's inspection parser against truncated command-specific payloads.
fn validate_cdo(words: &[u32]) -> Result<()> {
    if words.len() < 5 || words[..3] != [4, 0x004f4443, 0x200] {
        return Err("invalid CDO header".into());
    }
    let body = words
        .get(
            5..5usize
                .checked_add(words[3] as usize)
                .ok_or("CDO length overflow")?,
        )
        .ok_or("truncated CDO body")?;
    let mut i = 0;
    while i < body.len() {
        let header = body[i];
        i += 1;
        let id = header & 0xffff;
        let mut len = (header >> 16) as usize;
        if len == 255 {
            len = *body.get(i).ok_or("truncated long CDO header")? as usize;
            i += 1;
        }
        body.get(i..i.checked_add(len).ok_or("CDO command overflow")?)
            .ok_or("truncated CDO payload")?;
        if (id == cdo::CMD_WRITE && len != 2)
            || (id == cdo::CMD_MASK_WRITE && len != 3)
            || (id == cdo::CMD_DMA_WRITE && len < 2)
        {
            return Err("invalid CDO command length".into());
        }
        i += len;
    }
    Ok(())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Endpoint {
    tile: usize,
    port: u8,
}
#[derive(Clone, Copy)]
struct Task {
    bd: usize,
    repeat: usize,
    token: bool,
}
struct Active {
    engine: dma::Engine,
    task: Task,
}
struct Channel {
    tile: usize,
    index: usize,
    queue: VecDeque<Task>,
    active: Option<Active>,
    tokens: usize,
    paused: bool,
    endpoint: usize,
}
struct Core {
    tile: Tile,
    decoder: Option<PreparedDecoder>,
    enabled: bool,
    busy: u64,
    lock_wait: u64,
}
struct Memtile {
    memory: Vec<u8>,
    locks: [i32; 64],
}
/// Functional steps, not hardware cycles. Counts accumulate across submissions.
#[derive(Debug)]
pub struct CoreStats {
    pub col: u8,
    pub row: u8,
    pub busy: u64,
    pub lock_wait: u64,
}
#[derive(Debug)]
pub struct ShimStats {
    pub col: u8,
    pub direction: Direction,
    pub channel: u8,
    pub words: u64,
}
impl ShimStats {
    pub fn dram_bytes(&self) -> u64 {
        self.words * 4
    }
}
/// Array simulator loaded from the exact program/configuration bytes supplied
/// to CONFIG_CU and submit. Host arguments remain caller-owned mutable buffers.
pub struct Config {
    registers: BTreeMap<u32, u32>,
    cores: BTreeMap<usize, Core>,
    /// Ascending keys of `cores` (the stepping order), so `tick` can step a core while lending its neighbours.
    core_ids: Vec<usize>,
    memtiles: BTreeMap<usize, Memtile>,
    shim_locks: BTreeMap<usize, [i32; 16]>,
    channels: Vec<Channel>,
    bindings: BTreeMap<u32, usize>,
    shim_words: [[u64; 4]; 8],
    fifo_indices: BTreeMap<Endpoint, usize>,
    fifos: Vec<Fifo>,
    routes: Vec<(usize, Vec<usize>)>,
    routes_dirty: bool,
    /// Maximum ticks for each blocking SYNC/poll; exceeding this reports deadlock.
    pub tick_limit: usize,
    pub ticks: usize,
}
fn token_delivered(registers: &BTreeMap<u32, u32>, tile: usize, channel: usize) -> bool {
    let loc = location(tile);
    // Compute tokens are not routed through this shim TILE_CTRL endpoint.
    if loc.row != 0 {
        return false;
    }
    let reg = |off| registers.get(&loc.address(off)).copied().unwrap_or(0);
    let slave = reg(regs::shim::SS_SLAVE_TILE_CTRL);
    let master = reg(regs::shim::SS_MASTER_SOUTH0);
    if slave & (regs::SS_ENABLE | regs::SS_PACKET) != (regs::SS_ENABLE | regs::SS_PACKET)
        || master & (regs::SS_ENABLE | regs::SS_PACKET) != (regs::SS_ENABLE | regs::SS_PACKET)
    {
        return false;
    }
    let controller = (reg(channel_offset(0) + channel as u32 * 8) >> 8) & 31;
    (0..4).any(|slot| {
        let rule = reg(regs::shim::SS_SLAVE_TILE_CTRL_SLOT0 + slot * 4);
        let mask = (rule >> 16) & 31;
        rule & (1 << 8) != 0
            && controller & mask == ((rule >> 24) & 31) & mask
            && rule & 7 == master & 7
            && master & (1 << (3 + ((rule >> 4) & 3))) != 0
    })
}
fn location(index: usize) -> Location {
    Location::new((index / 6) as u32, (index % 6) as u32)
}
fn address(addr: u32) -> Result<(usize, u32)> {
    let col = addr >> 25;
    let row = (addr >> 20) & 31;
    if addr % 4 != 0 {
        return Err(format!("unaligned register address {addr:#x}"));
    }
    if col >= 8 || row >= 6 {
        return Err(format!("tile address outside array: {addr:#x}"));
    }
    Ok((col as usize * 6 + row as usize, addr & 0xfffff))
}
fn channel_offset(row: u32) -> u32 {
    match row {
        0 => 0x1d200,
        1 => 0xa0600,
        _ => 0x1de00,
    }
}
fn channel_count(tile: usize) -> usize {
    if tile % 6 == 1 {
        6
    } else {
        2
    }
}
fn bd_base(tile: usize) -> u32 {
    if tile % 6 == 1 {
        0xa0000
    } else {
        0x1d000
    }
}
fn bd_count(tile: usize) -> usize {
    if tile % 6 == 1 {
        48
    } else {
        16
    }
}
fn port(kind: TileType, index: u8, master: bool) -> Result<Port> {
    let table = match (kind, master) {
        (TileType::Compute, true) => &route::COMPUTE_MASTERS,
        (TileType::Compute, false) => &route::COMPUTE_SLAVES,
        (TileType::Memtile, true) => &route::MEMTILE_MASTERS,
        (TileType::Memtile, false) => &route::MEMTILE_SLAVES,
        (TileType::Shim, true) => &route::SHIM_MASTERS,
        (TileType::Shim, false) => &route::SHIM_SLAVES,
    };
    for (group, g) in table.iter().enumerate() {
        if index >= g.first && index - g.first < g.count {
            let n = index - g.first;
            return Ok(match group {
                0 => Port::Core(n),
                1 => Port::Dma(n),
                2 => Port::TileCtrl,
                3 => Port::Fifo(n),
                4 => Port::South(n),
                5 => Port::West(n),
                6 => Port::North(n),
                7 => Port::East(n),
                _ => Port::Trace(n),
            });
        }
    }
    Err("invalid switch port".into())
}
/// Private payload of `Config::state_snapshot`; derives give field-wise equality, never string comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ChannelState {
    tile: usize,
    index: usize,
    queue: Vec<(usize, usize, bool)>,
    active: Option<((usize, usize, bool), dma::EngineState)>,
    tokens: usize,
    paused: bool,
    endpoint: Endpoint,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct ConfigState {
    registers: BTreeMap<u32, u32>,
    bindings: BTreeMap<u32, usize>,
    core_locks: BTreeMap<usize, [i32; 16]>,
    memtile_locks: BTreeMap<usize, [i32; 64]>,
    shim_locks: BTreeMap<usize, [i32; 16]>,
    channels: Vec<ChannelState>,
    fifos: Vec<(Endpoint, usize, Vec<u32>)>,
}
impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}
fn lend(v: &mut Option<(Vec<u8>, [i32; 16])>) -> Option<Neighbour<'_>> {
    v.as_mut()
        .map(|(memory, locks)| Neighbour { memory, locks })
}
impl Config {
    pub fn new() -> Self {
        Self {
            registers: BTreeMap::new(),
            cores: BTreeMap::new(),
            core_ids: Vec::new(),
            memtiles: BTreeMap::new(),
            shim_locks: BTreeMap::new(),
            channels: Vec::new(),
            bindings: BTreeMap::new(),
            shim_words: [[0; 4]; 8],
            fifo_indices: BTreeMap::new(),
            fifos: Vec::new(),
            routes: Vec::new(),
            routes_dirty: true,
            tick_limit: 10_000_000,
            ticks: 0,
        }
    }
    pub fn from_pdi(bytes: &[u8]) -> Result<Self> {
        let words = pdi::extract_cdo(bytes).ok_or("invalid PDI")?;
        validate_cdo(&words)?;
        let cmds = cdo::parse(&words).ok_or("invalid CDO")?;
        let mut config = Self::new();
        config.load_cdo(&cmds)?;
        Ok(config)
    }
    pub fn load_cdo(&mut self, cmds: &[Cmd]) -> Result<()> {
        for cmd in cmds {
            match cmd {
                Cmd::Write(a, v) => self.write(*a, *v, u32::MAX)?,
                Cmd::MaskWrite(a, m, v) => self.write(*a, *v, *m)?,
                Cmd::DmaWrite(a, w) => {
                    let a = u32::try_from(*a).map_err(|_| "unsupported CDO address")?;
                    self.block_write(a, w)?;
                }
                Cmd::Nop(_) => {}
                Cmd::Other(id, _) => return Err(format!("unsupported CDO command {id:#x}")),
            }
        }
        Ok(())
    }
    fn core(&mut self, id: usize) -> Result<&mut Core> {
        if id % 6 < 2 {
            return Err("not a compute tile".into());
        }
        if !self.cores.contains_key(&id) {
            self.cores.insert(
                id,
                Core {
                    tile: Tile::new(Vec::new()).map_err(error)?,
                    decoder: None,
                    enabled: false,
                    busy: 0,
                    lock_wait: 0,
                },
            );
            if let Err(at) = self.core_ids.binary_search(&id) {
                self.core_ids.insert(at, id);
            }
        }
        Ok(self.cores.get_mut(&id).unwrap())
    }
    fn memtile(&mut self, id: usize) -> &mut Memtile {
        self.memtiles.entry(id).or_insert_with(|| Memtile {
            memory: vec![0; 512 * 1024],
            locks: [0; 64],
        })
    }
    pub fn core_stats(&self) -> impl Iterator<Item = CoreStats> + '_ {
        self.cores.iter().map(|(&id, c)| CoreStats {
            col: (id / 6) as u8,
            row: (id % 6) as u8,
            busy: c.busy,
            lock_wait: c.lock_wait,
        })
    }
    pub fn shim_stats(&self) -> impl Iterator<Item = ShimStats> + '_ {
        self.shim_words
            .iter()
            .enumerate()
            .flat_map(|(col, counts)| {
                counts
                    .iter()
                    .enumerate()
                    .map(move |(index, &words)| ShimStats {
                        col: col as u8,
                        direction: if index < 2 {
                            Direction::S2mm
                        } else {
                            Direction::Mm2s
                        },
                        channel: (index % 2) as u8,
                        words,
                    })
            })
    }
    fn endpoint(&mut self, key: Endpoint) -> usize {
        if let Some(&i) = self.fifo_indices.get(&key) {
            return i;
        }
        let i = self.fifos.len();
        self.fifos.push(Fifo::new(4));
        self.fifo_indices.insert(key, i);
        i
    }
    fn channel(&mut self, tile: usize, index: usize) -> Result<usize> {
        let count = channel_count(tile);
        if index >= count * 2 {
            return Err("invalid DMA channel".into());
        }
        if let Some(i) = self
            .channels
            .iter()
            .position(|c| c.tile == tile && c.index == index)
        {
            return Ok(i);
        }
        let loc = location(tile);
        let source = index >= count;
        let number = (index % count) as u8;
        let key = if source {
            let p = if loc.row == 0 {
                Port::South(if number == 0 { 3 } else { 7 })
            } else {
                Port::Dma(number)
            };
            Endpoint {
                tile,
                port: route::port_index(loc.kind(), p, false) as u8,
            }
        } else {
            Endpoint {
                tile,
                port: 32 + number,
            }
        };
        let endpoint = self.endpoint(key);
        let i = self.channels.len();
        self.channels.push(Channel {
            tile,
            index,
            queue: VecDeque::new(),
            active: None,
            tokens: 0,
            paused: false,
            endpoint,
        });
        Ok(i)
    }
    fn block_write(&mut self, addr: u32, words: &[u32]) -> Result<()> {
        for (i, &word) in words.iter().enumerate() {
            self.write(
                addr.checked_add(u32::try_from(i * 4).map_err(|_| "address overflow")?)
                    .ok_or("address overflow")?,
                word,
                u32::MAX,
            )?;
        }
        Ok(())
    }
    fn write(&mut self, addr: u32, value: u32, mask: u32) -> Result<()> {
        let (tile, off) = address(addr)?;
        let row = tile % 6;
        let old = self.read(addr)?;
        let value = (old & !mask) | (value & mask);
        if row == 1 && off < 512 * 1024 {
            self.memtile(tile).memory[off as usize..off as usize + 4]
                .copy_from_slice(&value.to_le_bytes());
            return Ok(());
        } else if row >= 2 && off < 65536 {
            let c = self.core(tile)?;
            let memory = c
                .tile
                .memory
                .get_mut(off as usize..off as usize + 4)
                .ok_or("data memory bounds")?;
            memory.copy_from_slice(&value.to_le_bytes());
            return Ok(());
        } else if row >= 2
            && (regs::core::PROGRAM_MEMORY..regs::core::PROGRAM_MEMORY + 16384).contains(&off)
        {
            let c = self.core(tile)?;
            let start = (off - regs::core::PROGRAM_MEMORY) as usize;
            if c.tile.program.len() < start + 4 {
                c.tile.program.resize(start + 4, 0)
            }
            c.tile.program[start..start + 4].copy_from_slice(&value.to_le_bytes());
            c.decoder = None;
            return Ok(());
        } else if row >= 2 && off == regs::core::CORE_CONTROL {
            if value & !3 != 0 {
                return Err("unsupported core control fields".into());
            }
            let c = self.core(tile)?;
            c.enabled = value & 1 != 0 && value & 2 == 0;
            if value & 2 != 0 {
                c.tile.reset_execution();
            }
        } else if row >= 2 && off == regs::core::CORE_PC {
            self.core(tile)?.tile.pc = value as usize;
        } else if row >= 2
            && (regs::core::LOCK0_VALUE..regs::core::LOCK0_VALUE + 256).contains(&off)
            && (off - regs::core::LOCK0_VALUE) % 16 == 0
        {
            if value > 63 {
                return Err("lock initialization exceeds six bits".into());
            }
            self.core(tile)?.tile.locks[((off - regs::core::LOCK0_VALUE) / 16) as usize] =
                value as i32;
        } else if row == 1 && (0xc0000..0xc0400).contains(&off) && (off - 0xc0000) % 16 == 0 {
            if value > 63 {
                return Err("lock initialization exceeds six bits".into());
            }
            self.memtile(tile).locks[((off - 0xc0000) / 16) as usize] = value as i32;
        } else if row == 0 && (0x14000..0x14100).contains(&off) && (off - 0x14000) % 16 == 0 {
            if value > 63 {
                return Err("lock initialization exceeds six bits".into());
            }
            self.shim_locks.entry(tile).or_insert([0; 16])[((off - 0x14000) / 16) as usize] =
                value as i32;
        } else if (bd_base(tile)..bd_base(tile) + bd_count(tile) as u32 * 32).contains(&off) {
            // Descriptor registers are latched when a queued/chained BD loads.
            if row >= 2 && (off - bd_base(tile)) % 32 >= 24 {
                return Err("reserved compute BD register".into());
            }
            if row >= 2 {
                self.core(tile)?;
            } else if row == 1 {
                self.memtile(tile);
            }
        } else if (channel_offset(row as u32)
            ..channel_offset(row as u32) + channel_count(tile) as u32 * 16)
            .contains(&off)
        {
            let offset = (off - channel_offset(row as u32)) as usize;
            let index = offset / 8;
            let ch = self.channel(tile, index)?;
            if offset % 8 == 4 {
                if value & !0x80ff003f != 0 {
                    return Err("unsupported DMA task fields".into());
                }
                if row != 0 && value >> 31 != 0 {
                    return Err("task completion tokens require modeled shim TILE_CTRL".into());
                }
                let bd = (value & 63) as usize;
                if bd >= bd_count(tile) || (row == 1 && (index % 6) % 2 != bd / 24) {
                    return Err("invalid DMA descriptor bank".into());
                }
                self.channels[ch].queue.push_back(Task {
                    bd,
                    repeat: ((value >> 16) & 255) as usize + 1,
                    token: value >> 31 != 0,
                });
            } else {
                let allowed = if row >= 2 {
                    0xff03
                } else if row == 1 {
                    0xff02
                } else {
                    0xff00
                };
                if value & !allowed != 0 {
                    return Err("unsupported DMA channel control fields".into());
                }
                if row != 0 && value & 2 != 0 {
                    self.channels[ch].active = None;
                    self.channels[ch].queue.clear();
                    self.channels[ch].tokens = 0;
                }
                // ENABLE is compute-only; memtile/shim queues start directly.
                if row >= 2 && mask & 1 != 0 {
                    self.channels[ch].paused = value & 1 == 0;
                }
                if row != 0 && mask & 2 != 0 && value & 2 == 0 && mask & 1 == 0 {
                    self.channels[ch].paused = false;
                }
                if row != 0 && value & 2 != 0 {
                    self.channels[ch].paused = true;
                }
            }
        } else if (row == 1 && (0xb0000..0xb0200).contains(&off))
            || (row != 1 && (0x3f000..0x3f200).contains(&off))
        {
            if value & (1 << 30) != 0
                && !(row == 0
                    && (off == regs::shim::SS_MASTER_SOUTH0
                        || off == regs::shim::SS_SLAVE_TILE_CTRL))
            {
                return Err("data packet switch mode not supported".into());
            }
            let base = if row == 1 { 0xb0000 } else { 0x3f000 };
            let master = off < base + 0x100;
            let p = port(
                location(tile).kind(),
                ((off - base - if master { 0 } else { 0x100 }) / 4) as u8,
                master,
            )?;
            let allowed = if value & regs::SS_PACKET != 0 {
                if master {
                    0xc00000ff
                } else {
                    0xc0000000
                }
            } else if master {
                0x8000007f
            } else {
                0x80000000
            };
            if value & !allowed != 0 {
                return Err("unsupported stream-switch fields".into());
            }
            if value & regs::SS_ENABLE != 0
                && !matches!(
                    p,
                    Port::Dma(_) | Port::South(_) | Port::North(_) | Port::West(_) | Port::East(_)
                )
                && !(row == 0 && p == Port::TileCtrl && value & regs::SS_PACKET != 0)
            {
                return Err("unsupported stream-switch endpoint".into());
            }
            self.routes_dirty = true;
        } else if row == 0 && (off == regs::shim::MUX_CONFIG || off == regs::shim::DEMUX_CONFIG) {
            // Only DMA0/1 selections are modeled; nonzero PLIO/other NOC
            // selections must not silently behave like DMA streams.
            let allowed = if off == regs::shim::MUX_CONFIG {
                0x4400
            } else {
                0x50
            };
            if value & !allowed != 0 {
                return Err("unsupported shim NOC mux selection".into());
            }
            self.routes_dirty = true;
        } else if row == 0
            && (regs::shim::SS_SLAVE_TILE_CTRL_SLOT0..regs::shim::SS_SLAVE_TILE_CTRL_SLOT0 + 16)
                .contains(&off)
        {
            // The packet matcher is evaluated when a shim completion emits its token.
            if value & !0x1f1f0137 != 0 {
                return Err("unsupported TILE_CTRL packet-rule fields".into());
            }
        } else {
            return Err(format!("unsupported register write {addr:#x}"));
        }
        self.registers.insert(addr, value);
        Ok(())
    }
    fn read(&self, addr: u32) -> Result<u32> {
        let (id, off) = address(addr)?;
        if id % 6 == 1 {
            if let Some(m) = self.memtiles.get(&id) {
                if off < 512 * 1024 {
                    return Ok(u32::from_le_bytes(
                        m.memory[off as usize..off as usize + 4].try_into().unwrap(),
                    ));
                }
                if (0xc0000..0xc0400).contains(&off) && (off - 0xc0000) % 16 == 0 {
                    return Ok(m.locks[((off - 0xc0000) / 16) as usize] as u32);
                }
            }
        }
        if id % 6 == 0 && (0x14000..0x14100).contains(&off) && (off - 0x14000) % 16 == 0 {
            return Ok(self
                .shim_locks
                .get(&id)
                .map(|l| l[((off - 0x14000) / 16) as usize] as u32)
                .unwrap_or(0));
        }
        if let Some(c) = self.cores.get(&id) {
            if off == regs::core::CORE_STATUS {
                return Ok(c.tile.status);
            }
            if off == regs::core::CORE_PC {
                return Ok(c.tile.pc as u32);
            }
            if (regs::core::LOCK0_VALUE..regs::core::LOCK0_VALUE + 256).contains(&off)
                && (off - regs::core::LOCK0_VALUE) % 16 == 0
            {
                return Ok(c.tile.locks[((off - regs::core::LOCK0_VALUE) / 16) as usize] as u32);
            }
            if off < 65536 {
                return Ok(u32::from_le_bytes(
                    c.tile.memory[off as usize..off as usize + 4]
                        .try_into()
                        .unwrap(),
                ));
            }
            if (regs::core::PROGRAM_MEMORY..regs::core::PROGRAM_MEMORY + 16384).contains(&off) {
                let a = (off - regs::core::PROGRAM_MEMORY) as usize;
                return Ok(c
                    .tile
                    .program
                    .get(a..a + 4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .unwrap_or(0));
            }
        }
        Ok(self.registers.get(&addr).copied().unwrap_or(0))
    }
    fn sink(&self, tile: usize, p: Port) -> Result<Option<Endpoint>> {
        let loc = location(tile);
        let dest = match p {
            Port::Dma(n) => return Ok(Some(Endpoint { tile, port: 32 + n })),
            Port::North(n) if loc.row < 5 => {
                Some((Location::new(loc.col, loc.row + 1), Port::South(n)))
            }
            Port::South(n) if loc.row > 0 => {
                Some((Location::new(loc.col, loc.row - 1), Port::North(n)))
            }
            Port::West(n) if loc.col > 0 => {
                Some((Location::new(loc.col - 1, loc.row), Port::East(n)))
            }
            Port::East(n) if loc.col < 7 => {
                Some((Location::new(loc.col + 1, loc.row), Port::West(n)))
            }
            Port::South(n) if loc.row == 0 && (n == 2 || n == 3) => {
                let channel = n - 2;
                let shift = 4 + 2 * channel as u32;
                let mux = self
                    .registers
                    .get(&loc.address(regs::shim::DEMUX_CONFIG))
                    .copied()
                    .unwrap_or(0);
                return Ok(if (mux >> shift) & 3 == 1 {
                    Some(Endpoint {
                        tile,
                        port: 32 + channel,
                    })
                } else {
                    None
                });
            }
            _ => None,
        };
        Ok(dest.map(|(loc, p)| Endpoint {
            tile: loc.col as usize * 6 + loc.row as usize,
            port: route::port_index(loc.kind(), p, false) as u8,
        }))
    }
    fn rebuild_routes(&mut self) -> Result<()> {
        let mut graph: BTreeMap<Endpoint, Vec<Endpoint>> = BTreeMap::new();
        for (&addr, &value) in &self.registers {
            let (tile, off) = address(addr)?;
            let loc = location(tile);
            let base = if loc.row == 1 { 0xb0000 } else { 0x3f000 };
            if !(base..base + 0x100).contains(&off)
                || value >> 31 == 0
                || value & regs::SS_PACKET != 0
            {
                continue;
            }
            let master = port(loc.kind(), ((off - base) / 4) as u8, true)?;
            let slave = (value & 127) as u8;
            port(loc.kind(), slave, false)?;
            let enable = self
                .registers
                .get(&loc.address(base + 0x100 + 4 * slave as u32))
                .copied()
                .unwrap_or(0);
            if enable >> 31 == 0 {
                continue;
            }
            if let Some(sink) = self.sink(tile, master)? {
                graph
                    .entry(Endpoint { tile, port: slave })
                    .or_default()
                    .push(sink);
            }
        }
        self.routes.clear();
        for (src, dests) in graph {
            let src = self.endpoint(src);
            let mut out = Vec::with_capacity(dests.len());
            for d in dests {
                let d = self.endpoint(d);
                if !out.contains(&d) {
                    out.push(d)
                }
            }
            self.routes.push((src, out));
        }
        self.routes_dirty = false;
        Ok(())
    }
    /// Step one enabled core with its South/West/North neighbour cores' data memory and locks
    /// lent to it for the step (memory is moved out and back, never copied; locks are 64 bytes).
    fn step_core(&mut self, id: usize) -> Result<()> {
        let c = self.cores.get_mut(&id).unwrap();
        if !c.enabled || c.tile.status & (1 << 20) != 0 {
            return Ok(());
        }
        if c.decoder.is_none() {
            c.decoder = Some(PreparedDecoder::new(&c.tile.program).map_err(error)?)
        }
        // Core tile ids are col*6+row; row 2 sits above a memtile, which is not a core memory view.
        let (col, row) = (id / 6, id % 6);
        let ids = [
            (row > 2).then(|| id - 1),
            (col > 0).then(|| id - 6),
            (row < 5).then(|| id + 1),
        ];
        let mut lent: [Option<(Vec<u8>, [i32; 16])>; 3] = [None, None, None];
        for (slot, n) in lent.iter_mut().zip(ids) {
            if let Some(n) = n.and_then(|n| self.cores.get_mut(&n)) {
                *slot = Some((std::mem::take(&mut n.tile.memory), n.tile.locks));
            }
        }
        let [mut south, mut west, mut north] = lent;
        let mut nb = Neighbours {
            south: lend(&mut south),
            west: lend(&mut west),
            north: lend(&mut north),
        };
        let c = self.cores.get_mut(&id).unwrap();
        let result = c.tile.step_with(c.decoder.as_ref().unwrap(), &mut nb);
        drop(nb);
        let pc = c.tile.pc;
        for (lent, n) in [south, west, north].into_iter().zip(ids) {
            if let (Some((memory, locks)), Some(n)) = (lent, n) {
                let t = &mut self.cores.get_mut(&n).unwrap().tile;
                t.memory = memory;
                t.locks = locks;
            }
        }
        let c = self.cores.get_mut(&id).unwrap();
        match result.map_err(|e| format!("core pc={pc:#x}: {e:?}"))? {
            Step::Blocked => c.lock_wait += 1,
            _ => c.busy += 1,
        }
        Ok(())
    }
    pub fn tick(&mut self, args: &mut [Vec<u8>]) -> Result<()> {
        if self.routes_dirty {
            self.rebuild_routes()?
        }
        for k in 0..self.core_ids.len() {
            let id = self.core_ids[k];
            self.step_core(id)?;
        }
        for i in 0..self.channels.len() {
            if self.channels[i].paused {
                continue;
            }
            if self.channels[i].active.is_none() {
                if let Some(task) = self.channels[i].queue.pop_front() {
                    self.channels[i].active = Some(Active {
                        engine: dma::Engine::new(task.bd, task.repeat).map_err(error)?,
                        task,
                    });
                }
            }
            let channel = &mut self.channels[i];
            let Some(active) = channel.active.as_mut() else {
                continue;
            };
            // Shim MM2S only injects when the corresponding NOC mux selects DMA.
            if channel.tile % 6 == 0 && channel.index >= 2 {
                let shift = if channel.index == 2 { 10 } else { 14 };
                let mux = self
                    .registers
                    .get(&location(channel.tile).address(regs::shim::MUX_CONFIG))
                    .copied()
                    .unwrap_or(0);
                if (mux >> shift) & 3 != 1 {
                    continue;
                }
            }
            let direction = if channel.index < channel_count(channel.tile) {
                Direction::S2mm
            } else {
                Direction::Mm2s
            };
            let mut access = ArrayAccess {
                registers: &mut self.registers,
                cores: &mut self.cores,
                memtiles: &mut self.memtiles,
                shim_locks: &mut self.shim_locks,
                bindings: &self.bindings,
                args,
                channel: channel.index,
            };
            let (result, transferred) = active
                .engine
                .step(
                    &mut access,
                    channel.tile,
                    direction,
                    &mut self.fifos[channel.endpoint],
                )
                .map_err(|e| {
                    format!("DMA tile={} channel={}: {e:?}", channel.tile, channel.index)
                })?;
            if channel.tile % 6 == 0 {
                self.shim_words[channel.tile / 6][channel.index] += transferred as u64;
            }
            if result == Step::Done {
                if active.task.token
                    && token_delivered(&self.registers, channel.tile, channel.index)
                {
                    channel.tokens += 1;
                }
                channel.active = None;
            }
        }
        for (src, dests) in &self.routes {
            if dests.is_empty()
                || self.fifos[*src].is_empty()
                || dests.iter().any(|d| !self.fifos[*d].has_space())
            {
                continue;
            }
            let value = self.fifos[*src].pop().unwrap();
            for &d in dests {
                assert!(self.fifos[d].push(value));
            }
        }
        self.ticks += 1;
        Ok(())
    }
    pub fn execute(&mut self, ops: &[TxnOp], args: &mut [Vec<u8>]) -> Result<()> {
        self.execute_with(ops, args, &mut |_, _| {})
    }
    /// `execute` with a host-side `producer(wait_tick,args)` called exactly once immediately before every simulator tick spent
    /// waiting inside MASKPOLL/SYNC. `wait_tick` starts at 0 for each call and increases monotonically across the whole command
    /// stream (it is not reset between ops), so a producer can publish operands/sequence numbers while the NPU is polling.
    pub fn execute_with(
        &mut self,
        ops: &[TxnOp],
        args: &mut [Vec<u8>],
        producer: &mut dyn FnMut(usize, &mut [Vec<u8>]),
    ) -> Result<()> {
        let mut wait_tick = 0usize;
        for op in ops {
            match *op {
                TxnOp::Write(a, v) => self.write(a, v, u32::MAX)?,
                TxnOp::MaskWrite(a, m, v) => self.write(a, v, m)?,
                TxnOp::BlockWrite(a, ref words) => self.block_write(a, words)?,
                TxnOp::DdrPatch {
                    address: addr,
                    arg,
                    plus,
                } => {
                    let (tile, off) = address(addr)?;
                    if tile % 6 != 0
                        || !(0x1d004..0x1d204).contains(&off)
                        || (off - 0x1d004) % 32 != 0
                        || arg >= args.len()
                        || plus % 4 != 0
                        || plus > args[arg].len() as u64
                    {
                        return Err("invalid DDR_PATCH binding".into());
                    }
                    self.bindings.insert(addr, arg);
                    // Virtual addresses identify args, while transfer engines use the bound slice.
                    let virtual_address = ((arg as u64 + 1) << 32) + plus;
                    self.write(addr, virtual_address as u32, u32::MAX)?;
                    let high = self.registers.get(&(addr + 4)).copied().unwrap_or(0);
                    self.write(
                        addr + 4,
                        (high & 0xffff0000) | ((virtual_address >> 32) as u32 & 0xffff),
                        u32::MAX,
                    )?;
                }
                TxnOp::MaskPoll(addr, mask, value) => {
                    let mut ready = false;
                    for tick in 0..=self.tick_limit {
                        if self.read(addr)? & mask == value & mask {
                            ready = true;
                            break;
                        }
                        if tick < self.tick_limit {
                            producer(wait_tick, args);
                            wait_tick += 1;
                            self.tick(args)?;
                        }
                    }
                    if !ready {
                        return Err(format!("MASKPOLL timed out at {addr:#x}"));
                    }
                }
                TxnOp::Sync {
                    col,
                    row,
                    direction,
                    channel,
                    cols,
                    rows,
                } => {
                    if cols == 0
                        || rows == 0
                        || col as u16 + cols as u16 > 8
                        || row as u16 + rows as u16 > 6
                        || direction > 1
                        || channel as usize >= channel_count(row as usize)
                    {
                        return Err("invalid SYNC rectangle/channel".into());
                    }
                    let mut targets = Vec::new();
                    for c in col..col + cols {
                        for r in row..row + rows {
                            let tile = c as usize * 6 + r as usize;
                            targets.push(self.channel(
                                tile,
                                direction as usize * channel_count(tile) + channel as usize,
                            )?);
                        }
                    }
                    let mut ready = false;
                    for tick in 0..=self.tick_limit {
                        if targets.iter().all(|&i| self.channels[i].tokens > 0) {
                            ready = true;
                            break;
                        }
                        if tick < self.tick_limit {
                            producer(wait_tick, args);
                            wait_tick += 1;
                            self.tick(args)?;
                        }
                    }
                    if !ready {
                        return Err(format!(
                            "SYNC timed out: col={col} row={row} dir={direction} channel={channel}"
                        ));
                    }
                    for i in targets {
                        self.channels[i].tokens -= 1;
                    }
                }
            }
        }
        Ok(())
    }
    pub fn submit(&mut self, bytes: &[u8], args: &mut [Vec<u8>]) -> Result<()> {
        self.execute(&parse_txn(bytes)?, args)
    }
    pub fn submit_with(
        &mut self,
        bytes: &[u8],
        args: &mut [Vec<u8>],
        producer: &mut dyn FnMut(usize, &mut [Vec<u8>]),
    ) -> Result<()> {
        self.execute_with(&parse_txn(bytes)?, args, producer)
    }
    /// Read-only retained-state fingerprint for `assert_eq!` across submits. Captures the register file
    /// (all BD/DMA/stream/mux/packet-rule words, including iteration-current updates), arg bindings, every
    /// core/memtile/shim lock, every DMA channel (queued tasks, tokens, paused, active task and the exact
    /// engine position: BD id, repeats, latched descriptor, walker, acquisition flag) and all stream FIFOs.
    /// Omits memories/host buffers, core execution state, and cumulative counters (`ticks`, stats).
    pub fn state_snapshot(&self) -> impl Eq + std::fmt::Debug {
        let task = |t: &Task| (t.bd, t.repeat, t.token);
        let mut channels: Vec<ChannelState> = self
            .channels
            .iter()
            .map(|c| ChannelState {
                tile: c.tile,
                index: c.index,
                queue: c.queue.iter().map(task).collect(),
                active: c.active.as_ref().map(|a| (task(&a.task), a.engine.state())),
                tokens: c.tokens,
                paused: c.paused,
                endpoint: self
                    .fifo_indices
                    .iter()
                    .find(|(_, &i)| i == c.endpoint)
                    .map(|(&k, _)| k)
                    .expect("channel endpoint is registered"),
            })
            .collect();
        channels.sort_by_key(|c| (c.tile, c.index));
        ConfigState {
            registers: self.registers.clone(),
            bindings: self.bindings.clone(),
            core_locks: self
                .cores
                .iter()
                .map(|(&id, c)| (id, c.tile.locks))
                .collect(),
            memtile_locks: self.memtiles.iter().map(|(&id, m)| (id, m.locks)).collect(),
            shim_locks: self.shim_locks.clone(),
            channels,
            fifos: self
                .fifo_indices
                .iter()
                .map(|(&k, &i)| {
                    let (capacity, words) = self.fifos[i].state();
                    (k, capacity, words)
                })
                .collect(),
        }
    }
}

struct ArrayAccess<'a> {
    registers: &'a mut BTreeMap<u32, u32>,
    cores: &'a mut BTreeMap<usize, Core>,
    memtiles: &'a mut BTreeMap<usize, Memtile>,
    shim_locks: &'a mut BTreeMap<usize, [i32; 16]>,
    bindings: &'a BTreeMap<u32, usize>,
    args: &'a mut [Vec<u8>],
    channel: usize,
}
impl ArrayAccess<'_> {
    fn host_arg(&self, tile: usize, id: usize) -> crate::sim::Result<usize> {
        self.bindings
            .get(&location(tile).address(bd_base(tile) + id as u32 * 32 + 4))
            .copied()
            .ok_or(crate::sim::Error::InvalidBd)
    }
    fn lock(&mut self, tile: usize, id: usize) -> crate::sim::Result<&mut i32> {
        match tile % 6 {
            0 => self
                .shim_locks
                .entry(tile)
                .or_insert([0; 16])
                .get_mut(id)
                .ok_or(crate::sim::Error::InvalidLock),
            1 => self
                .memtiles
                .get_mut(&tile)
                .ok_or(crate::sim::Error::Bounds)?
                .locks
                .get_mut(id.checked_sub(64).ok_or(crate::sim::Error::Unsupported)?)
                .ok_or(crate::sim::Error::InvalidLock),
            _ => self
                .cores
                .get_mut(&tile)
                .ok_or(crate::sim::Error::Bounds)?
                .tile
                .locks
                .get_mut(id)
                .ok_or(crate::sim::Error::InvalidLock),
        }
    }
    fn memory(
        &mut self,
        tile: usize,
        bd: &Descriptor,
        address: u64,
    ) -> crate::sim::Result<&mut [u8]> {
        let a = usize::try_from(address).map_err(|_| crate::sim::Error::Bounds)?;
        let (memory, a) = match tile % 6 {
            0 => {
                let arg = bd.host_arg.ok_or(crate::sim::Error::InvalidBd)?;
                (
                    self.args
                        .get_mut(arg)
                        .ok_or(crate::sim::Error::Bounds)?
                        .as_mut_slice(),
                    a,
                )
            }
            1 => {
                let a = a
                    .checked_sub(0x80000)
                    .ok_or(crate::sim::Error::Unsupported)?;
                (
                    self.memtiles
                        .get_mut(&tile)
                        .ok_or(crate::sim::Error::Bounds)?
                        .memory
                        .as_mut_slice(),
                    a,
                )
            }
            _ => (
                self.cores
                    .get_mut(&tile)
                    .ok_or(crate::sim::Error::Bounds)?
                    .tile
                    .memory
                    .as_mut_slice(),
                a,
            ),
        };
        memory
            .get_mut(a..a.checked_add(4).ok_or(crate::sim::Error::Bounds)?)
            .ok_or(crate::sim::Error::Bounds)
    }
}
impl Access for ArrayAccess<'_> {
    fn descriptor(&mut self, tile: usize, id: usize) -> crate::sim::Result<Descriptor> {
        if id >= bd_count(tile) || (tile % 6 == 1 && (self.channel % 6) % 2 != id / 24) {
            return Err(crate::sim::Error::InvalidBd);
        }
        let loc = location(tile);
        let mut words = [0; 8];
        for (i, w) in words.iter_mut().enumerate() {
            *w = self
                .registers
                .get(&loc.address(bd_base(tile) + id as u32 * 32 + i as u32 * 4))
                .copied()
                .unwrap_or(0)
        }
        let mut host_arg = None;
        if loc.row == 0 {
            let arg = self.host_arg(tile, id)?;
            host_arg = Some(arg);
            if words[2] & !0xffff != 0 {
                return Err(crate::sim::Error::Unsupported);
            }
            let address = (words[2] as u64) << 32 | words[1] as u64;
            let offset = address
                .checked_sub((arg as u64 + 1) << 32)
                .ok_or(crate::sim::Error::Bounds)?;
            words[1] = offset as u32;
            words[2] = (offset >> 32) as u32;
        }
        let mut bd = Descriptor::decode(loc.kind(), words)?;
        bd.host_arg = host_arg;
        Ok(bd)
    }
    fn iteration_loaded(
        &mut self,
        tile: usize,
        id: usize,
        bd: Descriptor,
    ) -> crate::sim::Result<()> {
        if bd.iteration.wrap == 1 {
            return Ok(());
        }
        let (word, shift) = match tile % 6 {
            0 => (6, 26),
            1 => (6, 23),
            _ => (4, 19),
        };
        let addr = location(tile).address(bd_base(tile) + id as u32 * 32 + word * 4);
        let old = self.registers.get(&addr).copied().unwrap_or(0);
        self.registers.insert(
            addr,
            (old & !(63 << shift))
                | ((((bd.iteration.current + 1) % bd.iteration.wrap) as u32) << shift),
        );
        Ok(())
    }
    fn acquire(&mut self, tile: usize, id: usize, value: i32) -> crate::sim::Result<bool> {
        let lock = self.lock(tile, id)?;
        if value < 0 {
            if *lock < -value {
                return Ok(false);
            }
            *lock += value;
            Ok(true)
        } else {
            Ok(*lock == value)
        }
    }
    fn release(&mut self, tile: usize, id: usize, value: i32) -> crate::sim::Result<()> {
        // Disabled/default release has value zero and must not dereference the
        // default memtile lock ID 0 (a neighboring lock view).
        if value != 0 {
            *self.lock(tile, id)? += value
        }
        Ok(())
    }
    fn read(&mut self, tile: usize, bd: &Descriptor, address: u64) -> crate::sim::Result<u32> {
        Ok(u32::from_le_bytes(
            self.memory(tile, bd, address)?.try_into().unwrap(),
        ))
    }
    fn write(
        &mut self,
        tile: usize,
        bd: &Descriptor,
        address: u64,
        value: u32,
    ) -> crate::sim::Result<()> {
        self.memory(tile, bd, address)?
            .copy_from_slice(&value.to_le_bytes());
        Ok(())
    }
}
