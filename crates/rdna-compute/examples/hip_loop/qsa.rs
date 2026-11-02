// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `qsa_hg4`: `indexed_attention_attention_q8_batched_hg4` (Q8 K/V, QSA prefill
//! attention) on gfx1100 at the Flash-Next S1 shape.
//!
//! Production launch reproduced here (all file:line in `rdna-compute/src/tensor_ops.rs`):
//! * route: `indexed_attention_attention_batch_launch` :9080; `hg4_bytes` :9146-9148
//!   (`!decode`, `!hg12` because format is Q8 not F32), kernel via
//!   `QsaKvFormat::Q8.kernel(..)` :9159-9166, grid `[n_heads / 4, 1, rows]` in
//!   WORKGROUPS (`QSA_ATTENTION_HG4_HEADS` :8859), block 256 (:140, :9211), dynamic
//!   LDS `qsa_attention_hg4_lds_bytes` :8880-8898 with `shape_selected =
//!   selected_capacity` (hipfire-dispatch `pipeline/layer_ops.rs:4580`).
//! * kernargs :9213-9239: q_with_gate, full_keys, full_values, selected, output
//!   (pointers), then i32 rows, position_start, n_heads, n_kv_heads, head_dim,
//!   budget_blocks, compress, capacity, full_capacity.
//! * shape (hipfire-arch-qwen4 `config.rs:1130-1152`): 24 q / 2 kv heads, head_dim
//!   256, indexer budget 2048 tokens, compress 4 -> `budget_blocks = budget /
//!   compress = 512` (`layer_ops.rs:4527`), `capacity = qsa_selected_capacity() =
//!   budget + compress - 1 = 2051` (`config.rs:1732`).
//! * selection (kernel `indexed_attention_select_f32_batched`, `kernels/src/
//!   tensor_ops.hip:4290-4435`, serial twin :4165-4215): per row `visible =
//!   position_start + row + 1`, `block_count = visible / compress`, `chosen =
//!   min(budget_blocks, block_count)`; the chosen blocks are written in SCORE-RANK
//!   order (score desc, block index asc), `compress` consecutive tokens each, then
//!   the incomplete tail tokens `block_count*compress .. visible`. Slots past
//!   `selected_len = min(chosen*compress + tail, capacity)` stay -1. The attention
//!   kernel derives the same `selected_len` itself (`tensor_ops.hip:5254-5259`).
//!   The scores are the indexer's; here they are seeded pseudo-random, so the
//!   ordering/address pattern is random-scatter over the causal prefix.
#![allow(dead_code)]

use crate::node::*;
use rdna_compute::Gpu;
use serde_json::{json, Value};

const SRC: &str = concat!(
    include_str!("../../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../../kernels/src/rope_yarn.h"),
    include_str!("../../../../kernels/src/tensor_ops.hip")
);
const SYMBOL: &str = "indexed_attention_attention_q8_batched_hg4";

const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const BUDGET_TOKENS: usize = 2048;
const COMPRESS: usize = 4;
const BUDGET_BLOCKS: usize = BUDGET_TOKENS / COMPRESS;
/// `qsa_selected_capacity()`: budget plus the incomplete tail.
const CAPACITY: usize = BUDGET_TOKENS + COMPRESS - 1;
const HG4_HEADS: usize = 4;
const BLOCK: u32 = 256;
const LDS_LIMIT: usize = 64 * 1024;
const TILE_ROWS: usize = 768;
/// FN S1 prefill length (22 tiles of 768); the K/V cache capacity.
const FN_TOKENS: usize = 22 * TILE_ROWS;
/// Pre-card emulator gate shape (`--shape emu`): 3 rows at `position_start` 102 -> visible 103, 104,
/// 105: every row far below the 2048-token budget (selected_len 103/104/105 < capacity, 25/26/26
/// blocks, tails 3/0/1: row 1 crosses a 4-token block boundary), full causal context (every visible
/// token selected), shape constants (24/2 heads, head_dim 256, budget 512 blocks, capacity 2051) as
/// production.
const EMU_POSITION: usize = 102;
const EMU_ROWS: usize = 3;
/// Q8_0 K or V bytes per token row: kv_heads * (head_dim / 32) * 34.
const KV_ROW_BYTES: usize = N_KV_HEADS * (HEAD_DIM / 32) * 34;
const Q_ROW_FLOATS: usize = 2 * N_HEADS * HEAD_DIM;
const OUT_ROW_FLOATS: usize = N_HEADS * HEAD_DIM;

/// `qsa_attention_hg4_lds_bytes` :8893-8897 (arch/route gates excluded).
fn production_lds(shape_selected: usize) -> Option<u32> {
    let bytes = shape_selected
        .checked_mul(4 * HG4_HEADS + 4)?
        .checked_add(4 * (HG4_HEADS * 256 + HG4_HEADS))?;
    (bytes <= LDS_LIMIT).then_some(bytes as u32)
}

/// Selected length of one row: the kernel's own derivation.
fn selected_len(position_start: usize, row: usize) -> usize {
    let visible = position_start + row + 1;
    let block_count = visible / COMPRESS;
    let chosen = BUDGET_BLOCKS.min(block_count);
    (chosen * COMPRESS + visible - block_count * COMPRESS).min(CAPACITY)
}

pub struct QsaHg4 {
    label: String,
    rows: usize,
    position_start: usize,
    full_capacity: usize,
    dev: Option<Dev>,
}

struct Dev {
    q: Buf,
    k: Buf,
    v: Buf,
    sel: Buf,
    out: Buf,
    pristine: Buf,
}

impl QsaHg4 {
    pub fn new(shape: &ShapeSpec) -> Result<Self> {
        let mut s = shape.clone();
        let mut pos = s.preset.take();
        if let Some(p) = s.kv.remove("pos") {
            if pos.is_some() {
                return Err("--shape: preset and pos= both given".into());
            }
            pos = Some(p);
        }
        let emu = pos.as_deref() == Some("emu");
        let rows = s.take_usize("rows", Some(if emu { EMU_ROWS } else { TILE_ROWS }))?;
        let cap = s.take_usize("cap", Some(0))?;
        s.no_leftover()?;
        // Tile 11 of 22 (mid-window) is the representative default, tile 21 the
        // late one; an integer is an explicit position_start; `emu` is the pre-card
        // emulator gate's small shape.
        let (label, position_start) = match pos.as_deref() {
            None | Some("mid") | Some("fn") => ("mid".to_owned(), 11 * TILE_ROWS),
            Some("late") => ("late".to_owned(), 21 * TILE_ROWS),
            Some("emu") => ("emu".to_owned(), EMU_POSITION),
            Some(n) => {
                let p = n.parse::<usize>().map_err(|e| {
                    format!("--shape pos={n}: expected mid|late|emu|<position_start>: {e}")
                })?;
                (format!("p{p}"), p)
            }
        };
        // Row-count gate of the grouped kernel (`QSA_ATTENTION_HG4_MIN_ROWS`, :8875).
        if rows < 2 {
            return Err(format!(
                "--shape rows={rows}: production routes < 2 rows off hg4 (decode policies)"
            ));
        }
        // The emulator gate maps the K/V cache exactly: no tokens past the last visible one.
        let full_capacity = if cap != 0 {
            if cap < position_start + rows {
                return Err(format!(
                    "--shape cap={cap} is below position_start + rows = {}",
                    position_start + rows
                ));
            }
            cap
        } else if emu {
            position_start + rows
        } else {
            (position_start + rows).max(FN_TOKENS)
        };
        Ok(Self {
            label,
            rows,
            position_start,
            full_capacity,
            dev: None,
        })
    }

    fn end_position(&self) -> usize {
        self.position_start + self.rows
    }

    fn lds_production(&self) -> u32 {
        production_lds(CAPACITY).expect("hg4 LDS for the FN selected capacity fits")
    }

    /// Sum over rows of the selected length.
    fn total_selected(&self) -> u64 {
        (0..self.rows)
            .map(|r| selected_len(self.position_start, r) as u64)
            .sum()
    }
}

impl Node for QsaHg4 {
    fn name(&self) -> &'static str {
        "qsa_hg4"
    }
    fn shape_label(&self) -> String {
        self.label.clone()
    }
    fn shape_json(&self) -> Value {
        json!({
            "n_heads": N_HEADS, "n_kv_heads": N_KV_HEADS, "head_dim": HEAD_DIM,
            "rows": self.rows, "position_start": self.position_start,
            "end_position": self.end_position(),
            "budget_tokens": BUDGET_TOKENS, "budget_blocks": BUDGET_BLOCKS,
            "compress": COMPRESS, "capacity": CAPACITY, "shape_selected": CAPACITY,
            "full_capacity": self.full_capacity, "kv_format": "q8",
            "kv_row_bytes": KV_ROW_BYTES,
            "selected_len_first_row": selected_len(self.position_start, 0),
            "selected_len_last_row": selected_len(self.position_start, self.rows - 1),
        })
    }

    fn incumbent(&self) -> Result<Incumbent> {
        Ok(Incumbent {
            module: "tensor_ops".into(),
            source: SRC.to_owned(),
            symbol: SYMBOL.into(),
            extra_flags: String::new(),
            selection: "crates/rdna-compute/src/tensor_ops.rs:9146-9176 (Q8 hg4 via QsaKvFormat::kernel; LDS :8880-8898); ensure_kernel_public(\"tensor_ops\", TENSOR_OPS_SRC) :9212".into(),
        })
    }

    fn geometry_with(&self, hdr: &mut Header) -> Result<Geometry> {
        let heads_per_wg = hdr.take_usize("heads_per_wg", HG4_HEADS)?;
        let rows_per_wg = hdr.take_usize("rows_per_wg", 1)?;
        let block = hdr.take_usize("block", BLOCK as usize)?;
        let lds = hdr.take_usize("lds_dynamic", self.lds_production() as usize)?;
        hdr.no_leftover()?;
        if heads_per_wg == 0 || N_HEADS % heads_per_wg != 0 {
            return Err(format!(
                "hip_loop header heads_per_wg={heads_per_wg} does not tile {N_HEADS} heads"
            ));
        }
        if rows_per_wg == 0 || self.rows % rows_per_wg != 0 {
            return Err(format!(
                "hip_loop header rows_per_wg={rows_per_wg} does not tile {} rows",
                self.rows
            ));
        }
        if block == 0 || block > 1024 {
            return Err(format!("hip_loop header block={block} outside 1..=1024"));
        }
        if lds > LDS_LIMIT {
            return Err(format!(
                "hip_loop header lds_dynamic={lds} exceeds {LDS_LIMIT}"
            ));
        }
        let g = Geometry {
            grid: [
                (N_HEADS / heads_per_wg) as u32,
                1,
                (self.rows / rows_per_wg) as u32,
            ],
            block: [block as u32, 1, 1],
            lds: lds as u32,
        };
        // Coverage of heads x rows.
        if g.grid[0] as usize * heads_per_wg != N_HEADS
            || g.grid[2] as usize * rows_per_wg != self.rows
        {
            return Err("hip_loop header geometry does not cover heads x rows".into());
        }
        Ok(g)
    }

    fn work(&self) -> Work {
        // QK and PV over the selected context: per (row, head) 2*hd FLOPs per
        // selected token for QK and the same for PV.
        let sel = self.total_selected() as f64;
        let flops = sel * N_HEADS as f64 * 4.0 * HEAD_DIM as f64;
        // Nominal DRAM-side traffic per workgroup (no cache reuse modelled): K and
        // V Q8 head slices (8 blocks x 34 B) per selected token, the index list
        // read twice (token pass, value pass), Q+gate of four heads, output of four.
        let head_slice = (HEAD_DIM / 32 * 34) as f64;
        let wgs_per_row = (N_HEADS / HG4_HEADS) as f64;
        let bytes = wgs_per_row
            * (sel * (2.0 * head_slice + 8.0)
                + self.rows as f64
                    * (HG4_HEADS * 2 * HEAD_DIM * 4 + HG4_HEADS * HEAD_DIM * 4) as f64);
        Work { flops, bytes }
    }

    fn record_extra(&self) -> Value {
        json!({
            "qsa": {
                "total_selected_tokens": self.total_selected(),
                "selection_order": "score-rank (seeded pseudo-random block scores; score desc, block asc), 4 tokens/block, then incomplete tail",
                "q_dtype": "f32 [rows][n_heads][q256, gate256]",
                "kv": "Q8_0 blocks (f16 scale + 32 int8), 34 B x 8 per head, 2 heads",
                "work_note": "flops = 4*hd*heads*sum(selected_len); bytes nominal, no cache reuse",
            }
        })
    }

    fn extent_shapes(&self) -> Vec<String> {
        // Judge shapes at the FN capacity (16896 tokens), then the K/V capacity at max_seq scale: the
        // late window of a 262144-token cache, and a 16-row window at its far end (cache populated to 261016).
        [
            "pos=mid",
            "pos=late",
            "pos=late,cap=262144",
            "pos=261000,rows=16,cap=262144",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect()
    }
    fn out_owner<'a>(&'a self, hdr: &Header) -> Result<Box<dyn Fn(usize) -> [u32; 3] + 'a>> {
        // out is f32 [rows][n_heads][head_dim]; the workgroup is (head group, 0, row group).
        let get = |k: &str, d: usize| -> Result<usize> {
            match hdr.kv.get(k) {
                Some(v) => v
                    .parse::<usize>()
                    .map_err(|e| format!("hip_loop header {k}={v}: {e}")),
                None => Ok(d),
            }
        };
        let (hpw, rpw) = (get("heads_per_wg", HG4_HEADS)?, get("rows_per_wg", 1)?);
        if hpw == 0 || rpw == 0 {
            return Err("hip_loop header: heads_per_wg/rows_per_wg must be positive".into());
        }
        Ok(Box::new(move |e| {
            let (row, head) = (e / OUT_ROW_FLOATS, e % OUT_ROW_FLOATS / HEAD_DIM);
            [(head / hpw) as u32, 0, (row / rpw) as u32]
        }))
    }
    fn emu_case(&self, seed: u64) -> Result<EmuCase> {
        let inputs = generate(self.rows, self.position_start, self.full_capacity, seed);
        validate_selection(
            &inputs.sel,
            self.rows,
            self.position_start,
            self.full_capacity,
        )?;
        let out_bytes = self.rows * OUT_ROW_FLOATS * 4;
        let mut args = vec![
            EmuArg::Buf(0),
            EmuArg::Buf(1),
            EmuArg::Buf(2),
            EmuArg::Buf(3),
            EmuArg::Buf(4),
        ];
        for v in [
            self.rows,
            self.position_start,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            BUDGET_BLOCKS,
            COMPRESS,
            CAPACITY,
            self.full_capacity,
        ] {
            args.push(EmuArg::I32(i32::try_from(v).map_err(err)?));
        }
        Ok(EmuCase {
            bufs: vec![
                EmuBuf {
                    name: "q",
                    bytes: inputs.q,
                },
                EmuBuf {
                    name: "k",
                    bytes: inputs.k,
                },
                EmuBuf {
                    name: "v",
                    bytes: inputs.v,
                },
                EmuBuf {
                    name: "sel",
                    bytes: bytemuck_i32(&inputs.sel).to_vec(),
                },
                // NaN-filled like the card's pristine output.
                EmuBuf {
                    name: "out",
                    bytes: vec![0xFFu8; out_bytes],
                },
            ],
            args,
            out: 4,
        })
    }

    fn setup(&mut self, gpu: &Gpu, seed: u64) -> Result<()> {
        if self.dev.is_some() {
            return Err("qsa_hg4: setup called twice".into());
        }
        if self.lds_production() as usize > LDS_LIMIT {
            return Err("qsa_hg4: LDS over the device limit".into());
        }
        let inputs = generate(self.rows, self.position_start, self.full_capacity, seed);
        validate_selection(
            &inputs.sel,
            self.rows,
            self.position_start,
            self.full_capacity,
        )?;
        let q = Buf::from_bytes(gpu, &inputs.q)?;
        let k = Buf::from_bytes(gpu, &inputs.k)?;
        let v = Buf::from_bytes(gpu, &inputs.v)?;
        let sel = Buf::from_bytes(gpu, bytemuck_i32(&inputs.sel))?;
        let out_bytes = self.rows * OUT_ROW_FLOATS * 4;
        let out = Buf::alloc(gpu, out_bytes)?;
        // NaN-filled pristine output: any element a launch fails to write shows up
        // as a byte difference only if the other arm wrote it.
        let pristine = Buf::from_bytes(gpu, &vec![0xFFu8; out_bytes])?;
        out.copy_from(gpu, &pristine)?;
        self.dev = Some(Dev {
            q,
            k,
            v,
            sel,
            out,
            pristine,
        });
        Ok(())
    }

    fn reset(&mut self, gpu: &Gpu) -> Result<()> {
        let d = self.dev.as_ref().ok_or("qsa_hg4: reset before setup")?;
        d.out.copy_from(gpu, &d.pristine)
    }

    fn kernarg(&self) -> Result<Vec<u8>> {
        let d = self.dev.as_ref().ok_or("qsa_hg4: kernarg before setup")?;
        let mut a = Args::new();
        a.ptr(&d.q).ptr(&d.k).ptr(&d.v).ptr(&d.sel).ptr(&d.out);
        for v in [
            self.rows,
            self.position_start,
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            BUDGET_BLOCKS,
            COMPRESS,
            CAPACITY,
            self.full_capacity,
        ] {
            a.i32(i32::try_from(v).map_err(err)?);
        }
        Ok(a.finish())
    }

    fn read_output(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        self.dev
            .as_ref()
            .ok_or("qsa_hg4: read before setup")?
            .out
            .download(gpu)
    }

    fn teardown(&mut self, gpu: &Gpu) {
        if let Some(d) = self.dev.take() {
            d.q.free(gpu);
            d.k.free(gpu);
            d.v.free(gpu);
            d.sel.free(gpu);
            d.out.free(gpu);
            d.pristine.free(gpu);
        }
    }
}

fn bytemuck_i32(v: &[i32]) -> &[u8] {
    // SAFETY: i32 has no padding and any alignment is valid for u8.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

pub struct Inputs {
    /// f32 [rows][n_heads][q(256), gate(256)] little-endian bytes.
    pub q: Vec<u8>,
    /// Q8_0 K cache, `full_capacity` token rows.
    pub k: Vec<u8>,
    pub v: Vec<u8>,
    /// i32 [rows][CAPACITY].
    pub sel: Vec<i32>,
}

fn fill_q8(rng: &mut Xs64, tokens: usize, populated: usize) -> Vec<u8> {
    let mut out = vec![0u8; tokens * KV_ROW_BYTES];
    for row in out.chunks_exact_mut(KV_ROW_BYTES).take(populated) {
        for block in row.chunks_exact_mut(34) {
            // Finite positive f16 scale in [~0.004, ~0.02] (codes up to 127 -> |x| < 2.6).
            let scale = 0.004 + 0.016 * (rng.unit() * 0.5 + 0.5);
            block[..2].copy_from_slice(&f16_bits(scale).to_le_bytes());
            for c in &mut block[2..] {
                // -127..=127 (never -128, matching a symmetric quantiser).
                *c = (((rng.next_u64() >> 33) % 255) as i64 - 127) as i8 as u8;
            }
        }
    }
    out
}

/// Seeded valid inputs. Tokens `>= position_start + rows` stay zero (never read:
/// the kernel masks `token < visible`).
pub fn generate(rows: usize, position_start: usize, full_capacity: usize, seed: u64) -> Inputs {
    let end = position_start + rows;
    let mut rq = Xs64::new(seed ^ 0x51);
    let mut q = Vec::with_capacity(rows * Q_ROW_FLOATS * 4);
    for _ in 0..rows * N_HEADS {
        for d in 0..2 * HEAD_DIM {
            // Query half in [-1, 1); gate half spread wider so sigmoid is non-trivial.
            let x = rq.unit() * if d < HEAD_DIM { 1.0 } else { 2.0 };
            q.extend_from_slice(&x.to_le_bytes());
        }
    }
    let k = fill_q8(&mut Xs64::new(seed ^ 0x4B), full_capacity, end);
    let v = fill_q8(&mut Xs64::new(seed ^ 0x56), full_capacity, end);
    let mut sel = vec![-1i32; rows * CAPACITY];
    let mut order: Vec<(u32, u32)> = Vec::new();
    for r in 0..rows {
        let mut rs = Xs64::new(seed ^ (0x5E1_0000 + r as u64 * 0x9E37_79B9));
        let visible = position_start + r + 1;
        let block_count = visible / COMPRESS;
        let chosen = BUDGET_BLOCKS.min(block_count);
        order.clear();
        // 20-bit scores: ties occur, resolved by ascending block index like the kernel.
        order.extend((0..block_count).map(|b| ((rs.next_u64() >> 44) as u32, b as u32)));
        order.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let row_sel = &mut sel[r * CAPACITY..(r + 1) * CAPACITY];
        for (rank, &(_, block)) in order.iter().take(chosen).enumerate() {
            for s in 0..COMPRESS {
                let (idx, tok) = (rank * COMPRESS + s, block as usize * COMPRESS + s);
                if idx < CAPACITY && tok < visible {
                    row_sel[idx] = tok as i32;
                }
            }
        }
        let mut off = chosen * COMPRESS;
        for tok in block_count * COMPRESS..visible {
            if off < CAPACITY {
                row_sel[off] = tok as i32;
                off += 1;
            }
        }
    }
    Inputs { q, k, v, sel }
}

/// Selection validity: each row fills exactly `selected_len` leading slots with
/// distinct in-range causal tokens (`0 <= t < visible`, `t < full_capacity`) in
/// whole `COMPRESS`-token blocks followed by the contiguous tail; the rest is -1.
pub fn validate_selection(
    sel: &[i32],
    rows: usize,
    position_start: usize,
    full_capacity: usize,
) -> Result<()> {
    if sel.len() != rows * CAPACITY {
        return Err("selection length mismatch".into());
    }
    let mut seen = vec![u32::MAX; full_capacity];
    for r in 0..rows {
        let visible = position_start + r + 1;
        let block_count = visible / COMPRESS;
        let chosen = BUDGET_BLOCKS.min(block_count);
        let sl = selected_len(position_start, r);
        let row = &sel[r * CAPACITY..(r + 1) * CAPACITY];
        for (i, &t) in row.iter().enumerate() {
            if i >= sl {
                if t != -1 {
                    return Err(format!(
                        "row {r} slot {i}: expected -1 past selected_len {sl}"
                    ));
                }
                continue;
            }
            if t < 0 || t as usize >= visible || t as usize >= full_capacity {
                return Err(format!("row {r} slot {i}: token {t} outside 0..{visible}"));
            }
            if seen[t as usize] == r as u32 {
                return Err(format!("row {r}: token {t} selected twice"));
            }
            seen[t as usize] = r as u32;
            if i < chosen * COMPRESS {
                let base = row[i - i % COMPRESS];
                if base as usize % COMPRESS != 0 || t != base + (i % COMPRESS) as i32 {
                    return Err(format!(
                        "row {r} slot {i}: not a whole {COMPRESS}-token block"
                    ));
                }
            } else if t as usize != block_count * COMPRESS + (i - chosen * COMPRESS) {
                return Err(format!("row {r} slot {i}: tail token {t} out of order"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(shape: &str) -> QsaHg4 {
        QsaHg4::new(&ShapeSpec::parse(Some(shape)).unwrap()).unwrap()
    }

    #[test]
    fn production_grid_and_lds() {
        let n = node("");
        let g = n.geometry();
        // grid in workgroups: 24 heads / 4 per WG x 768 rows (PF-1 threads (1536,1,768)).
        assert_eq!(g.grid, [6, 1, 768]);
        assert_eq!(g.block, [256, 1, 1]);
        assert_eq!(g.threads(), [1536, 1, 768]);
        // 2051 * 20 + 4 * (4*256 + 4) = 41020 + 4112
        assert_eq!(g.lds, 45132);
        assert_eq!(production_lds(CAPACITY), Some(45132));
        assert_eq!(CAPACITY, 2051);
        assert_eq!(BUDGET_BLOCKS, 512);
        assert_eq!(KV_ROW_BYTES, 544);
        assert_eq!(node("late").geometry(), g);
    }

    #[test]
    fn production_source_matches_reproduction() {
        let ops = include_str!("../../src/tensor_ops.rs");
        for needle in [
            "const QSA_ATTENTION_HG4_HEADS: usize = 4;",
            // The non-lite (shipped-before-batch-2) LDS formula this node reproduces; beta's q8 gfx1100 default
            // is now the hg4_lite twin (`selected * 16 + 128`), which this node does not model yet.
            "4 * QSA_ATTENTION_HG4_HEADS + 4,",
            "4 * (QSA_ATTENTION_HG4_HEADS * 256 + QSA_ATTENTION_HG4_HEADS),",
            "p.n_heads / QSA_ATTENTION_HG4_HEADS,",
            "const QSA_ATTENTION_PARALLEL_THREADS: u32 = 256;",
            "const QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES: usize = 64 * 1024;",
            "\"indexed_attention_attention_q8_batched_hg4\",",
        ] {
            assert!(
                ops.contains(needle),
                "tensor_ops.rs no longer contains `{needle}`"
            );
        }
        assert!(SRC.contains("indexed_attention_attention_##TAG##_batched_hg4"));
        assert!(SRC.contains("QSA_ATTENTION_KERNELS(q8, QSA_KV_Q8, unsigned char)"));
        let cfg = include_str!("../../../hipfire-arch-qwen4/src/config.rs");
        assert!(cfg.contains("self.indexer_budget + self.indexer_compress_ratio.saturating_sub(1)"));
        assert!(cfg.contains("expect(self.indexer_budget, 2048, \"indexer_budget\")?;"));
        assert!(
            cfg.contains("expect(self.indexer_compress_ratio, 4, \"indexer_compress_ratio\")?;")
        );
    }

    #[test]
    fn positions_and_shape_keys() {
        assert_eq!(node("").position_start, 8448);
        assert_eq!(node("pos=mid").position_start, 8448);
        assert_eq!(node("pos=late").position_start, 16128);
        assert_eq!(node("pos=late").end_position(), FN_TOKENS);
        assert_eq!(node("pos=1000,rows=16").position_start, 1000);
        assert!(QsaHg4::new(&ShapeSpec::parse(Some("pos=x")).unwrap()).is_err());
        assert!(QsaHg4::new(&ShapeSpec::parse(Some("bogus=1")).unwrap()).is_err());
        assert!(QsaHg4::new(&ShapeSpec::parse(Some("rows=1")).unwrap()).is_err());
        // Mid window: every row selects the full 2048 + tail (0..=3) tokens.
        for p in [8448usize, 16128] {
            for r in [0usize, 1, 2, 3, 4, 767] {
                let sl = selected_len(p, r);
                let tail = (p + r + 1) % COMPRESS;
                assert_eq!(sl, 2048 + tail);
            }
        }
    }

    #[test]
    fn header_geometry_refusals() {
        let n = node("");
        let mut h = Header::parse("// hip_loop: heads_per_wg=8 block=128\n").unwrap();
        let g = n.geometry_with(&mut h).unwrap();
        assert_eq!(g.grid, [3, 1, 768]);
        assert_eq!(g.block, [128, 1, 1]);
        assert_eq!(g.lds, 45132);
        let mut h = Header::parse("// hip_loop: rows_per_wg=2 lds_dynamic=1024\n").unwrap();
        let g = n.geometry_with(&mut h).unwrap();
        assert_eq!((g.grid, g.lds), ([6, 1, 384], 1024));
        for bad in [
            "heads_per_wg=5",
            "heads_per_wg=0",
            "rows_per_wg=5",
            "block=0",
            "lds_dynamic=70000",
            "nope=1",
        ] {
            let mut h = Header::parse(&format!("// hip_loop: {bad}\n")).unwrap();
            assert!(n.geometry_with(&mut h).is_err(), "{bad}");
        }
    }

    #[test]
    fn selection_valid_and_full_budget() {
        for (p, rows) in [(8448usize, 24usize), (16128, 24), (16128 + 744, 24)] {
            let full = (p + rows).max(FN_TOKENS);
            let inp = generate(rows, p, full, 7);
            validate_selection(&inp.sel, rows, p, full).unwrap();
            for r in 0..rows {
                let row = &inp.sel[r * CAPACITY..(r + 1) * CAPACITY];
                let live = row.iter().filter(|&&t| t >= 0).count();
                assert_eq!(live, selected_len(p, r));
                assert!(live >= 2048);
            }
            assert_eq!(inp.q.len(), rows * Q_ROW_FLOATS * 4);
            assert_eq!(inp.k.len(), full * KV_ROW_BYTES);
        }
        // Determinism and seed sensitivity.
        let a = generate(4, 8448, FN_TOKENS, 1);
        let b = generate(4, 8448, FN_TOKENS, 1);
        let c = generate(4, 8448, FN_TOKENS, 2);
        assert!(a.sel == b.sel && a.q == b.q && a.k == b.k);
        assert!(a.sel != c.sel);
    }

    #[test]
    fn selection_is_score_rank_not_sorted_and_corruption_is_caught() {
        let mut inp = generate(2, 8448, FN_TOKENS, 3);
        let row = &inp.sel[..CAPACITY];
        assert!(
            row[..2048].windows(2).any(|w| w[1] < w[0]),
            "rank order, not ascending"
        );
        inp.sel[5] = 1 << 30;
        assert!(validate_selection(&inp.sel, 2, 8448, FN_TOKENS).is_err());
        let mut inp = generate(2, 8448, FN_TOKENS, 3);
        inp.sel[0] = 8448 + 5; // acausal for row 0 (visible = 8449)
        assert!(validate_selection(&inp.sel, 2, 8448, FN_TOKENS).is_err());
    }

    #[test]
    fn q8_scales_finite_and_work_estimate() {
        let inp = generate(2, 64, FN_TOKENS, 9);
        for row in inp.k.chunks_exact(KV_ROW_BYTES).take(66) {
            for b in row.chunks_exact(34) {
                let s = half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32();
                assert!(s.is_finite() && s > 0.0);
            }
        }
        let w = node("").work();
        // 768 rows x ~2049.5 selected x 24 heads x 4 x 256 FLOPs.
        let sel = node("").total_selected() as f64;
        assert_eq!(w.flops, sel * 24.0 * 1024.0);
        assert!(w.flops > 3.8e10 && w.flops < 3.9e10, "{}", w.flops);
        assert!(w.bytes > 0.0);
    }

    #[test]
    fn incumbent_is_tensor_ops() {
        let i = node("").incumbent().unwrap();
        assert_eq!(i.module, "tensor_ops");
        assert_eq!(i.symbol, SYMBOL);
        assert!(i.source.contains(SYMBOL) || i.source.contains("_batched_hg4"));
    }
}
