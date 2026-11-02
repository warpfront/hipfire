// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Offline GEMM stream harness using the kernel's actual encoded program and
//! descriptors, not a direct matrix-multiply shortcut.
use crate::sim::{
    decode::PreparedDecoder,
    dma::{Direction, Fifo, ShimDma, TileDma},
    Error, Result, Step, Tile,
};
use crate::{
    dma::shim_bd,
    kernels::gemm_i8::{self, PackedJob},
};
/// Concatenate packed microtile jobs, run DMA rings and return C jobs in order.
pub fn simulate(k: usize, jobs: &[PackedJob]) -> Result<Vec<u8>> {
    if jobs.is_empty() || jobs.len() > gemm_i8::MAX_JOBS || ![64, 128, 256].contains(&k) {
        return Err(Error::Bounds);
    }
    if jobs
        .iter()
        .any(|j| j.a.len() != 8 * k || j.b.len() != 8 * k)
    {
        return Err(Error::Bounds);
    }
    let mut tile = Tile::new(gemm_i8::core_program(k, jobs.len()).finish())?;
    let decoder = PreparedDecoder::new(&tile.program)?;
    tile.locks = gemm_i8::initial_locks();
    let descriptors = gemm_i8::tile_bds();
    let mut input_dma = TileDma::new(Direction::S2mm);
    let mut c_dma = TileDma::new(Direction::Mm2s);
    for (id, bd) in descriptors.into_iter().enumerate() {
        input_dma.descriptors[id] = Some(bd);
        c_dma.descriptors[id] = Some(bd);
    }
    input_dma.start(0)?;
    c_dma.start(4)?;
    let mut input = Vec::with_capacity(jobs.len() * 16 * k);
    for j in jobs {
        for (a, b) in j.a.chunks_exact(512).zip(j.b.chunks_exact(512)) {
            input.extend_from_slice(a);
            input.extend_from_slice(b);
        }
    }
    let mut c = vec![0; 256 * jobs.len()];
    let mut input_shim = ShimDma::new(Direction::Mm2s, shim_bd((input.len() / 4) as u32, 0))?;
    let mut c_shim = ShimDma::new(Direction::S2mm, shim_bd((c.len() / 4) as u32, 0))?;
    let mut incoming = Fifo::new(4);
    let mut cf = Fifo::new(4);
    for _ in 0..jobs.len() * k * 64 + 10000 {
        let core = tile.step(&decoder)?;
        input_shim.step(&mut input, &mut incoming)?;
        input_dma.step(&mut tile, &mut incoming)?;
        c_dma.step(&mut tile, &mut cf)?;
        let output = c_shim.step(&mut c, &mut cf)?;
        if core == Step::Done && output == Step::Done {
            return Ok(c);
        }
    }
    Err(Error::Limit)
}
