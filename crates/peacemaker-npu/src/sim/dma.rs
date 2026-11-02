// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Functional compute/memtile/shim DMA. Descriptor fields follow peacemaker-npu::dma
//! (aie-rt MIT); dimensions are inner-first word strides. FIFO backpressure
//! never consumes a word, lock, iteration or descriptor completion twice.
use crate::dma::{ShimAxi, TileType};
use crate::sim::{Error, Result, Step, Tile};
use std::collections::VecDeque;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Mm2s,
    S2mm,
}
pub struct Fifo {
    words: VecDeque<u32>,
    capacity: usize,
}
impl Fifo {
    pub fn new(capacity: usize) -> Self {
        Self {
            words: VecDeque::with_capacity(capacity),
            capacity,
        }
    }
    pub fn push(&mut self, v: u32) -> bool {
        if !self.has_space() {
            false
        } else {
            self.words.push_back(v);
            true
        }
    }
    pub fn pop(&mut self) -> Option<u32> {
        self.words.pop_front()
    }
    pub fn len(&self) -> usize {
        self.words.len()
    }
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }
    pub(crate) fn has_space(&self) -> bool {
        self.words.len() < self.capacity
    }
    pub fn route(&mut self, dst: &mut Self) -> Step {
        if self.words.is_empty() || !dst.has_space() {
            Step::Blocked
        } else {
            dst.words.push_back(self.words.pop_front().unwrap());
            Step::Advanced
        }
    }
    /// Read-only `(capacity, queued words oldest-first)` for state comparison.
    pub(crate) fn state(&self) -> (usize, Vec<u32>) {
        (self.capacity, self.words.iter().copied().collect())
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dim {
    step: u64,
    wrap: u32,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Iteration {
    pub step: u32,
    pub wrap: u8,
    pub current: u8,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Descriptor {
    pub address: u64,
    pub words: u32,
    dims: [Dim; 4],
    pub iteration: Iteration,
    pub acq: Option<(usize, i32)>,
    pub rel: (usize, i32),
    pub next: Option<usize>,
    pub host_arg: Option<usize>,
}
fn s7(v: u32) -> i32 {
    ((v << 25) as i32) >> 25
}
fn iteration(word: u32, width: u32) -> Result<Iteration> {
    let current = ((word >> (width + 6)) & 63) as u8;
    let wrap = (((word >> width) & 63) + 1) as u8;
    if current >= wrap || (word as u64) >> (width + 12) != 0 {
        return Err(Error::InvalidBd);
    }
    Ok(Iteration {
        step: (word & ((1 << width) - 1)) + 1,
        wrap,
        current,
    })
}
fn tail(word: u32) -> (Option<(usize, i32)>, (usize, i32), Option<usize>) {
    (
        if word & (1 << 12) != 0 {
            Some(((word & 15) as usize, s7((word >> 5) & 127)))
        } else {
            None
        },
        (((word >> 13) & 15) as usize, s7((word >> 18) & 127)),
        if word & (1 << 26) != 0 {
            Some(((word >> 27) & 15) as usize)
        } else {
            None
        },
    )
}
impl Descriptor {
    pub(crate) fn decode(kind: TileType, w: [u32; 8]) -> Result<Self> {
        let default = Dim { step: 1, wrap: 0 };
        let (address, words, dims, iter, acq, rel, next) = match kind {
            TileType::Compute => {
                if w[5] & (1 << 25) == 0 {
                    return Err(Error::InvalidBd);
                }
                if w[1] != 0 || w[0] >> 28 != 0 || w[2] >> 26 != 0 || w[3] >> 29 != 0 {
                    return Err(Error::Unsupported);
                }
                let (acq, rel, next) = tail(w[5]);
                (
                    ((w[0] >> 14) & 0x3fff) as u64 * 4,
                    w[0] & 0x3fff,
                    [
                        Dim {
                            step: ((w[2] & 0x1fff) + 1) as u64,
                            wrap: (w[3] >> 13) & 255,
                        },
                        Dim {
                            step: (((w[2] >> 13) & 0x1fff) + 1) as u64,
                            wrap: (w[3] >> 21) & 255,
                        },
                        Dim {
                            step: ((w[3] & 0x1fff) + 1) as u64,
                            wrap: 0,
                        },
                        default,
                    ],
                    iteration(w[4], 13)?,
                    acq,
                    rel,
                    next,
                )
            }
            TileType::Memtile => {
                if w[7] >> 31 == 0 {
                    return Err(Error::InvalidBd);
                }
                if w[0] & !0x1ffff != 0
                    || w[1] >> 26 != 0
                    || w[2] & 0x78000000 != 0
                    || w[3] >> 27 != 0
                    || w[4] >> 27 != 0
                    || w[5] >> 17 != 0
                {
                    return Err(Error::Unsupported);
                }
                let next = if w[1] & (1 << 19) != 0 {
                    Some(((w[1] >> 20) & 63) as usize)
                } else {
                    None
                };
                if next.is_some_and(|id| id >= 48) {
                    return Err(Error::InvalidBd);
                }
                let dims = [
                    Dim {
                        step: ((w[2] & 0x1ffff) + 1) as u64,
                        wrap: (w[2] >> 17) & 1023,
                    },
                    Dim {
                        step: ((w[3] & 0x1ffff) + 1) as u64,
                        wrap: (w[3] >> 17) & 1023,
                    },
                    Dim {
                        step: ((w[4] & 0x1ffff) + 1) as u64,
                        wrap: (w[4] >> 17) & 1023,
                    },
                    Dim {
                        step: ((w[5] & 0x1ffff) + 1) as u64,
                        wrap: 0,
                    },
                ];
                (
                    (w[1] & 0x7ffff) as u64 * 4,
                    w[0] & 0x1ffff,
                    dims,
                    iteration(w[6], 17)?,
                    if w[7] & (1 << 15) != 0 {
                        Some(((w[7] & 255) as usize, s7((w[7] >> 8) & 127)))
                    } else {
                        None
                    },
                    (((w[7] >> 16) & 255) as usize, s7((w[7] >> 24) & 127)),
                    next,
                )
            }
            TileType::Shim => {
                if w[7] & (1 << 25) == 0 {
                    return Err(Error::InvalidBd);
                }
                if w[1] & 3 != 0 {
                    return Err(Error::Alignment);
                }
                // Burst length (w4 30..31), AxQoS (w5 20..23) and AxCACHE (w5 24..27) shape AXI traffic, not data: accepted.
                if w[2] & !0xffff != 0 || w[3] >> 30 != 0 || w[5] >> 28 != 0 {
                    return Err(Error::Unsupported);
                }
                let (acq, rel, next) = tail(w[7]);
                (
                    ((w[2] as u64) << 32) | w[1] as u64,
                    w[0],
                    [
                        Dim {
                            step: ((w[3] & 0xfffff) + 1) as u64,
                            wrap: (w[3] >> 20) & 1023,
                        },
                        Dim {
                            step: ((w[4] & 0xfffff) + 1) as u64,
                            wrap: (w[4] >> 20) & 1023,
                        },
                        Dim {
                            step: ((w[5] & 0xfffff) + 1) as u64,
                            wrap: 0,
                        },
                        default,
                    ],
                    iteration(w[6], 20)?,
                    acq,
                    rel,
                    next,
                )
            }
        };
        Ok(Self {
            address,
            words,
            dims,
            iteration: iter,
            acq,
            rel,
            next,
            host_arg: None,
        })
    }
}
/// Incremental tensor address generation avoids division/modulo per word.
struct Walker {
    coordinates: [u32; 4],
    offset: u64,
    base: u64,
    remaining: u32,
}
impl Walker {
    fn new(bd: &Descriptor) -> Self {
        Self {
            coordinates: [0; 4],
            offset: 0,
            base: bd.address + bd.iteration.step as u64 * bd.iteration.current as u64 * 4,
            remaining: bd.words,
        }
    }
    fn address(&self) -> u64 {
        self.base + self.offset * 4
    }
    fn advance(&mut self, dims: &[Dim; 4]) {
        self.remaining -= 1;
        if self.remaining == 0 {
            return;
        }
        for i in 0..4 {
            self.offset += dims[i].step;
            if dims[i].wrap == 0 {
                break;
            }
            self.coordinates[i] += 1;
            if self.coordinates[i] < dims[i].wrap {
                break;
            }
            self.offset -= dims[i].wrap as u64 * dims[i].step;
            self.coordinates[i] = 0;
        }
    }
}
/// Config's register/memory view, or a stand-alone tile/host view.
pub(crate) trait Access {
    fn descriptor(&mut self, tile: usize, id: usize) -> Result<Descriptor>;
    fn iteration_loaded(&mut self, tile: usize, id: usize, bd: Descriptor) -> Result<()>;
    fn acquire(&mut self, tile: usize, id: usize, value: i32) -> Result<bool>;
    fn release(&mut self, tile: usize, id: usize, value: i32) -> Result<()>;
    fn read(&mut self, tile: usize, bd: &Descriptor, address: u64) -> Result<u32>;
    fn write(&mut self, tile: usize, bd: &Descriptor, address: u64, value: u32) -> Result<()>;
}
struct Transfer {
    bd: Descriptor,
    walker: Walker,
    acquired: bool,
}
pub(crate) struct Engine {
    start: usize,
    id: usize,
    repeat: usize,
    current: Option<Transfer>,
    done: bool,
}
impl Engine {
    pub(crate) fn new(id: usize, repeat: usize) -> Result<Self> {
        if repeat == 0 || repeat > 256 {
            return Err(Error::InvalidBd);
        }
        Ok(Self {
            start: id,
            id,
            repeat,
            current: None,
            done: false,
        })
    }
    /// The boolean reports an actual transferred word (including the final
    /// word); an empty BD can complete without increasing DRAM counters.
    pub(crate) fn step<A: Access>(
        &mut self,
        access: &mut A,
        tile: usize,
        direction: Direction,
        fifo: &mut Fifo,
    ) -> Result<(Step, bool)> {
        if self.done {
            return Ok((Step::Done, false));
        }
        if self.current.is_none() {
            let bd = access.descriptor(tile, self.id)?;
            let walker = Walker::new(&bd);
            self.current = Some(Transfer {
                bd,
                walker,
                acquired: false,
            });
        }
        let transfer = self.current.as_mut().unwrap();
        if !transfer.acquired {
            if let Some((id, value)) = transfer.bd.acq {
                if !access.acquire(tile, id, value)? {
                    return Ok((Step::Blocked, false));
                }
            }
            access.iteration_loaded(tile, self.id, transfer.bd)?;
            transfer.acquired = true;
        }
        let transferred = transfer.walker.remaining > 0;
        if transferred {
            let address = transfer.walker.address();
            match direction {
                Direction::Mm2s => {
                    if !fifo.has_space() {
                        return Ok((Step::Blocked, false));
                    }
                    let value = access.read(tile, &transfer.bd, address)?;
                    assert!(fifo.push(value));
                }
                Direction::S2mm => {
                    let Some(value) = fifo.pop() else {
                        return Ok((Step::Blocked, false));
                    };
                    access.write(tile, &transfer.bd, address, value)?;
                }
            }
            transfer.walker.advance(&transfer.bd.dims);
        }
        if transfer.walker.remaining == 0 {
            let bd = transfer.bd;
            access.release(tile, bd.rel.0, bd.rel.1)?;
            if let Some(next) = bd.next {
                self.id = next
            } else if self.repeat > 1 {
                self.repeat -= 1;
                self.id = self.start
            } else {
                self.done = true
            }
            self.current = None;
        }
        Ok((
            if self.done {
                Step::Done
            } else {
                Step::Advanced
            },
            transferred,
        ))
    }
}
/// Read-only copy of every retained engine field for state comparison: the
/// queue-visible task, current BD id, remaining repeats, and, if a BD is latched,
/// its decoded descriptor, walker position and lock-acquisition flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EngineState {
    pub start: usize,
    pub id: usize,
    pub repeat: usize,
    pub done: bool,
    pub current: Option<TransferState>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TransferState {
    pub address: u64,
    pub words: u32,
    pub dims: [(u64, u32); 4],
    pub iteration: (u32, u8, u8),
    pub acq: Option<(usize, i32)>,
    pub rel: (usize, i32),
    pub next: Option<usize>,
    pub host_arg: Option<usize>,
    pub coordinates: [u32; 4],
    pub offset: u64,
    pub base: u64,
    pub remaining: u32,
    pub acquired: bool,
}
impl Engine {
    pub(crate) fn state(&self) -> EngineState {
        EngineState {
            start: self.start,
            id: self.id,
            repeat: self.repeat,
            done: self.done,
            current: self.current.as_ref().map(|t| {
                let b = &t.bd;
                let w = &t.walker;
                TransferState {
                    address: b.address,
                    words: b.words,
                    dims: b.dims.map(|d| (d.step, d.wrap)),
                    iteration: (b.iteration.step, b.iteration.wrap, b.iteration.current),
                    acq: b.acq,
                    rel: b.rel,
                    next: b.next,
                    host_arg: b.host_arg,
                    coordinates: w.coordinates,
                    offset: w.offset,
                    base: w.base,
                    remaining: w.remaining,
                    acquired: t.acquired,
                }
            }),
        }
    }
}

/// Stand-alone compute DMA convenience API; Config uses the same engine with
/// an array-aware register and memory view.
pub struct TileDma {
    pub descriptors: [Option<[u32; 6]>; 16],
    pub direction: Direction,
    engine: Option<Engine>,
}
struct TileAccess<'a> {
    tile: &'a mut Tile,
    descriptors: &'a mut [Option<[u32; 6]>; 16],
}
impl Access for TileAccess<'_> {
    fn descriptor(&mut self, _: usize, id: usize) -> Result<Descriptor> {
        let w = self
            .descriptors
            .get(id)
            .and_then(|v| *v)
            .ok_or(Error::InvalidBd)?;
        let mut words = [0; 8];
        words[..6].copy_from_slice(&w);
        Descriptor::decode(TileType::Compute, words)
    }
    fn iteration_loaded(&mut self, _: usize, id: usize, bd: Descriptor) -> Result<()> {
        let w = self.descriptors[id].as_mut().unwrap();
        w[4] = (w[4] & !(63 << 19))
            | ((((bd.iteration.current + 1) % bd.iteration.wrap) as u32) << 19);
        Ok(())
    }
    fn acquire(&mut self, _: usize, id: usize, value: i32) -> Result<bool> {
        self.tile.acquire(id, value)
    }
    fn release(&mut self, _: usize, id: usize, value: i32) -> Result<()> {
        self.tile.release(id, value)
    }
    fn read(&mut self, _: usize, _: &Descriptor, address: u64) -> Result<u32> {
        let a = usize::try_from(address).map_err(|_| Error::Bounds)?;
        Ok(u32::from_le_bytes(
            self.tile
                .memory
                .get(a..a + 4)
                .ok_or(Error::Bounds)?
                .try_into()
                .unwrap(),
        ))
    }
    fn write(&mut self, _: usize, _: &Descriptor, address: u64, value: u32) -> Result<()> {
        let a = usize::try_from(address).map_err(|_| Error::Bounds)?;
        self.tile
            .memory
            .get_mut(a..a + 4)
            .ok_or(Error::Bounds)?
            .copy_from_slice(&value.to_le_bytes());
        Ok(())
    }
}
impl TileDma {
    pub fn new(direction: Direction) -> Self {
        Self {
            descriptors: [None; 16],
            direction,
            engine: None,
        }
    }
    pub fn start(&mut self, id: usize) -> Result<()> {
        if self.engine.as_ref().is_some_and(|e| !e.done) {
            return Err(Error::InvalidBd);
        }
        let mut words = [0; 8];
        words[..6].copy_from_slice(
            &self
                .descriptors
                .get(id)
                .and_then(|v| *v)
                .ok_or(Error::InvalidBd)?,
        );
        Descriptor::decode(TileType::Compute, words)?;
        self.engine = Some(Engine::new(id, 1)?);
        Ok(())
    }
    pub fn step(&mut self, tile: &mut Tile, fifo: &mut Fifo) -> Result<Step> {
        let Some(engine) = self.engine.as_mut() else {
            return Ok(Step::Done);
        };
        engine
            .step(
                &mut TileAccess {
                    tile,
                    descriptors: &mut self.descriptors,
                },
                0,
                self.direction,
                fifo,
            )
            .map(|(state, _)| state)
    }
}
pub struct ShimDma {
    pub direction: Direction,
    descriptor: [u32; 8],
    engine: Engine,
}
struct HostAccess<'a> {
    host: &'a mut [u8],
    descriptor: &'a mut [u32; 8],
}
impl Access for HostAccess<'_> {
    fn descriptor(&mut self, _: usize, id: usize) -> Result<Descriptor> {
        if id != 0 {
            return Err(Error::InvalidBd);
        }
        Descriptor::decode(TileType::Shim, *self.descriptor)
    }
    fn iteration_loaded(&mut self, _: usize, _: usize, bd: Descriptor) -> Result<()> {
        self.descriptor[6] = (self.descriptor[6] & !(63 << 26))
            | ((((bd.iteration.current + 1) % bd.iteration.wrap) as u32) << 26);
        Ok(())
    }
    fn acquire(&mut self, _: usize, _: usize, _: i32) -> Result<bool> {
        Err(Error::Unsupported)
    }
    fn release(&mut self, _: usize, _: usize, value: i32) -> Result<()> {
        if value == 0 {
            Ok(())
        } else {
            Err(Error::Unsupported)
        }
    }
    fn read(&mut self, _: usize, _: &Descriptor, address: u64) -> Result<u32> {
        let a = usize::try_from(address).map_err(|_| Error::Bounds)?;
        Ok(u32::from_le_bytes(
            self.host
                .get(a..a + 4)
                .ok_or(Error::Bounds)?
                .try_into()
                .unwrap(),
        ))
    }
    fn write(&mut self, _: usize, _: &Descriptor, address: u64, value: u32) -> Result<()> {
        let a = usize::try_from(address).map_err(|_| Error::Bounds)?;
        self.host
            .get_mut(a..a + 4)
            .ok_or(Error::Bounds)?
            .copy_from_slice(&value.to_le_bytes());
        Ok(())
    }
}
impl ShimDma {
    pub fn new(direction: Direction, bd: [u32; 8]) -> Result<Self> {
        Descriptor::decode(TileType::Shim, bd)?;
        Ok(Self {
            direction,
            descriptor: bd,
            engine: Engine::new(0, 1)?,
        })
    }
    /// AXI attributes latched in the BD words (`BDi_4` burst 30..31, `BDi_5` AxCACHE 24..27 and AxQoS 20..23).
    /// Read back for verification only: the functional simulator has no AXI/timing model, so these never affect
    /// data movement. Non-zero SMID and SecureAccess are rejected by [`ShimDma::new`] (unsupported), not reported here.
    pub fn axi(&self) -> ShimAxi {
        ShimAxi {
            burst: self.descriptor[4] >> 30,
            cache: (self.descriptor[5] >> 24) & 15,
            qos: (self.descriptor[5] >> 20) & 15,
        }
    }
    pub fn step(&mut self, host: &mut [u8], fifo: &mut Fifo) -> Result<Step> {
        self.engine
            .step(
                &mut HostAccess {
                    host,
                    descriptor: &mut self.descriptor,
                },
                0,
                self.direction,
                fifo,
            )
            .map(|(state, _)| state)
    }
}
