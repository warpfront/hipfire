// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! DPU transaction stream (the per-command "instruction buffer" the NPU firmware executes).
//!
//! Layout = firmware TXN v0.1 as emitted by mlir-aie (`include/aie/Runtime/TxnEncoding.h`):
//! 16-byte header `{major=0, minor=1, dev_gen, rows, cols, memtile_rows, pad16, num_ops u32,
//! size_bytes u32}` then ops (all little-endian u32 words):
//! - WRITE      `[0, 0, addr, 0, value, 24]`
//! - BLOCKWRITE `[1, col | row << 8, addr, 16 + 4n, data...]`
//! - MASKWRITE  `[3, 0, addr, 0, value, mask, 28]`
//! - MASKPOLL   `[4, 0, addr, 0, value, mask, 28]`
//! - TCT sync   `[0x80, 16, dir | row << 8 | col << 16, nrow << 8 | ncol << 16 | chan << 24]`
//! - DDR_PATCH  `[0x81, 48, 0, 0, 0, 0, addr, 0, arg_idx, 0, plus_lo, plus_hi]`
//! Addresses are full array addresses (col/row folded in, partition-relative columns).

pub const DEV_GEN_AIE2P: u8 = 4;

pub struct Txn {
    dev_gen: u8,
    rows: u8,
    cols: u8,
    mem_rows: u8,
    ops: u32,
    words: Vec<u32>,
}

impl Txn {
    pub fn new(dev_gen: u8, rows: u8, cols: u8, mem_rows: u8) -> Txn {
        Txn {
            dev_gen,
            rows,
            cols,
            mem_rows,
            ops: 0,
            words: Vec::new(),
        }
    }
    /// Strix / Strix Halo NPU (npu2/npu4/npu5 class): 6 rows, 8 columns, 1 mem-tile row.
    pub fn aie2p_8col() -> Txn {
        Txn::new(DEV_GEN_AIE2P, 6, 8, 1)
    }

    /// Words of the 16-byte header: the first op word sits at u32 index `HEADER_WORDS` of [`Txn::to_bytes`].
    pub const HEADER_WORDS: usize = 4;
    /// Word offset of `value` inside a `write32` / `mask_poll` op (`[0,0,addr,0,value,24]`, `[4,0,addr,0,value,mask,28]`).
    pub const WRITE32_VALUE_WORD: usize = 4;
    pub const MASKPOLL_VALUE_WORD: usize = 4;

    /// Number of op payload words appended so far (excludes the header).
    pub fn word_len(&self) -> usize {
        self.words.len()
    }
    /// Number of ops appended so far.
    pub fn op_count(&self) -> usize {
        self.ops as usize
    }
    /// u32 index inside [`Txn::to_bytes`] of the first word of the NEXT op to be appended. Combined with
    /// [`Txn::WRITE32_VALUE_WORD`] / [`Txn::MASKPOLL_VALUE_WORD`] it names a value word a host may patch.
    pub fn next_op_word(&self) -> usize {
        Self::HEADER_WORDS + self.words.len()
    }
    /// Append every op of `other` (its header geometry is ignored; both streams must target the same device).
    pub fn append(&mut self, other: &Txn) -> &mut Self {
        self.words.extend_from_slice(&other.words);
        self.ops += other.ops;
        self
    }

    pub fn write32(&mut self, addr: u32, value: u32) -> &mut Self {
        self.words.extend([0, 0, addr, 0, value, 24]);
        self.ops += 1;
        self
    }

    pub fn block_write(&mut self, addr: u32, col: u32, row: u32, data: &[u32]) -> &mut Self {
        self.words.extend([
            1,
            (col & 0xff) | ((row & 0xff) << 8),
            addr,
            (16 + 4 * data.len()) as u32,
        ]);
        self.words.extend_from_slice(data);
        self.ops += 1;
        self
    }

    pub fn mask_write(&mut self, addr: u32, value: u32, mask: u32) -> &mut Self {
        self.words.extend([3, 0, addr, 0, value, mask, 28]);
        self.ops += 1;
        self
    }

    pub fn mask_poll(&mut self, addr: u32, value: u32, mask: u32) -> &mut Self {
        self.words.extend([4, 0, addr, 0, value, mask, 28]);
        self.ops += 1;
        self
    }

    /// Wait for the task-complete token of (col, row, dir: 0 = S2MM / 1 = MM2S, chan).
    pub fn sync(
        &mut self,
        col: u32,
        row: u32,
        dir: u32,
        chan: u32,
        ncol: u32,
        nrow: u32,
    ) -> &mut Self {
        self.words.extend([
            0x80,
            16,
            (dir & 0xff) | ((row & 0xff) << 8) | ((col & 0xff) << 16),
            ((nrow & 0xff) << 8) | ((ncol & 0xff) << 16) | ((chan & 0xff) << 24),
        ]);
        self.ops += 1;
        self
    }

    /// Patch `addr` (a shim BD address-low register) with buffer argument `arg_idx` + `plus`.
    pub fn ddr_patch(&mut self, addr: u32, arg_idx: u32, plus: u64) -> &mut Self {
        self.words.extend([
            0x81,
            48,
            0,
            0,
            0,
            0,
            addr,
            0,
            arg_idx,
            0,
            plus as u32,
            (plus >> 32) as u32,
        ]);
        self.ops += 1;
        self
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let size = 16 + 4 * self.words.len() as u32;
        let mut out = Vec::with_capacity(size as usize);
        out.extend([
            0u8,
            1,
            self.dev_gen,
            self.rows,
            self.cols,
            self.mem_rows,
            0,
            0,
        ]);
        out.extend(self.ops.to_le_bytes());
        out.extend(size.to_le_bytes());
        for w in &self.words {
            out.extend(w.to_le_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dma, regs};
    /// Rebuild the vendor add_one runtime sequence (mlir-aie npu2) op by op; must be byte-identical.
    #[test]
    fn reproduces_vendor_add_one_insts() {
        let Some(corpus) = crate::vendor_corpus() else {
            return;
        };
        let p = corpus.join("8309a7e39620419663f2aff6/insts.bin");
        let Ok(orig) = std::fs::read(&p) else {
            eprintln!("skip: {} missing", p.display());
            return;
        };
        let bd = dma::shim_bd(4096, 0);
        let mut t = Txn::aie2p_8col();
        t.block_write(regs::shim::DMA_BD0, 0, 0, &bd)
            .ddr_patch(regs::shim::DMA_BD0 + 4, 0, 0)
            .write32(regs::shim::DMA_MM2S_0_TASK_QUEUE, 0)
            .block_write(regs::shim::DMA_BD0 + 0x20, 0, 0, &bd)
            .ddr_patch(regs::shim::DMA_BD0 + 0x24, 0, 0)
            .mask_write(regs::shim::DMA_S2MM_0_CTRL, 0xf00, 0x1f00)
            .write32(regs::shim::DMA_S2MM_0_TASK_QUEUE, 0x8000_0001)
            .sync(0, 0, 0, 0, 1, 1);
        assert_eq!(t.to_bytes(), orig);
    }
}
