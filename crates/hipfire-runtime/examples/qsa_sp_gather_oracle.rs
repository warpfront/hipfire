// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle (and bench) of the sequence-parallel QSA selection on two gfx1201 cards: each rank
//! selects only its own contiguous rows of a prefill tile, packs the chosen block ids into a
//! compact `[rows][budget_blocks]` i32 buffer, all-gathers the compact rows with
//! `ArHcExchange::sp_ag_rows_lockstep` (`SpAgElem::Bf16`, `row_elems = 2 * budget_blocks`, DENSE wire) and
//! expands the peer rows back; the result must be byte-identical to the full-tile select.
//!
//! ```text
//! qsa_sp_gather_oracle [--devices 0,1] [--out DIR] [--bench]
//! ```
//!
//! Cases `(P, n)` (tile start position, rows): (0, 4096), (28672, 4096), (32768, 3830) and
//! (8192, 4096) with a tie-heavy pooled table (8 distinct pooled rows repeated), each on the
//! work-balanced split; (0, 4096) again on the even split. Per rank and case:
//!  (i) the selected rows (all n) equal the full-tile mirrored select run on the same card;
//!  (ii) the persisted last-row selection equals the full-tile one (rank 1: its own mirrored
//!       output; rank 0: row n-1 of the expanded buffer).
//! `--bench`: median of 50 iterations of pack + `sp_ag_rows_lockstep` + expand at (0, 4096) and
//! (32768, 4096), plus the full-tile select vs the own-range select per rank, so the report
//! carries the expected per-layer saving.
//! Prints `PASS|FAIL first_diff=row,slot` per case and rank; `ORACLE PASS` and exit 0 only if
//! every case passes (exit 1 otherwise, exit 3 on a fatal rank error). The same text lands in
//! `<out>/report.txt`.

use hipfire_runtime::ar_hc_exchange::{ArHcExchange, ArHcMode};
use rdna_compute::ar_hc_write::{SpAgElem, SP_ROW_CAP};
use rdna_compute::qsa_sp_select::QsaSpBlocks;
use rdna_compute::tensor_ops::{
    indexed_attention_select_batch, indexed_attention_select_batch_mirrored,
    IndexedAttentionSelectBatch,
};
use rdna_compute::Gpu;
use rdna_compute::{DType, GpuTensor};
use std::sync::{Arc, Barrier};
use std::time::Instant;

const INDEX_HEADS: usize = 4;
const INDEX_DIM: usize = 128;
/// `(index_heads + index_kv_heads) * index_dim`: the query row stride (the kv columns are unused).
const QUERY_STRIDE: usize = 640;
const BUDGET_BLOCKS: usize = 512;
const COMPRESS: usize = 4;
const CAPACITY: usize = 2051;
const BENCH_ITERS: usize = 50;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        let mut r = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        for _ in 0..4 {
            r.next();
        }
        r
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-1, 1).
    fn signed(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn es<E: ToString>(e: E) -> String {
    e.to_string()
}

fn median_us(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Cost of tile row `r` (its visible-key count): `P + r + 1`.
fn row_cost(position_start: usize, r: usize) -> usize {
    position_start + r + 1
}

/// Work-balanced split: the smallest `s` in `[1, n-1]` with `left(s) >= right(s)`, replaced by
/// `s-1` when that leaves the smaller maximum side (ties keep the smaller `s`).
fn balanced_split(n: usize, position_start: usize) -> usize {
    assert!(n >= 2);
    let total: usize = (0..n).map(|r| row_cost(position_start, r)).sum();
    let left = |s: usize| -> usize { (0..s).map(|r| row_cost(position_start, r)).sum() };
    let worst = |s: usize| -> usize {
        let l = left(s);
        l.max(total - l)
    };
    let mut acc = 0usize;
    let mut s = n - 1;
    for cand in 1..n {
        acc += row_cost(position_start, cand - 1);
        if acc >= total - acc {
            s = cand;
            break;
        }
    }
    if s > 1 && worst(s - 1) <= worst(s) {
        s -= 1;
    }
    s
}

#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    position_start: usize,
    rows: usize,
    tie_heavy: bool,
    even_split: bool,
}

const CASES: [Case; 5] = [
    Case {
        name: "p0_n4096",
        position_start: 0,
        rows: 4096,
        tie_heavy: false,
        even_split: false,
    },
    Case {
        name: "p28672_n4096",
        position_start: 28672,
        rows: 4096,
        tie_heavy: false,
        even_split: false,
    },
    Case {
        name: "p32768_n3830",
        position_start: 32768,
        rows: 3830,
        tie_heavy: false,
        even_split: false,
    },
    Case {
        name: "p8192_n4096_ties",
        position_start: 8192,
        rows: 4096,
        tie_heavy: true,
        even_split: false,
    },
    Case {
        name: "p0_n4096_even",
        position_start: 0,
        rows: 4096,
        tie_heavy: false,
        even_split: true,
    },
];

const BENCH_CASES: [(usize, usize); 2] = [(0, 4096), (32768, 4096)];

fn bf16_bits(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Seeded query (`[n][640]` f32) and pooled (`[pooled_capacity][128]` bf16 bytes). Identical on
/// both ranks. `tie_heavy` repeats 8 distinct pooled rows over the whole table.
fn make_data(case: &Case, pooled_capacity: usize) -> (Vec<f32>, Vec<u8>) {
    let mut rng = Rng::new(0x51a5_0000 ^ ((case.position_start as u64) << 20) ^ case.rows as u64);
    let query: Vec<f32> = (0..case.rows * QUERY_STRIDE)
        .map(|_| rng.signed())
        .collect();
    let mut pooled_rows: Vec<u16> = Vec::with_capacity(pooled_capacity * INDEX_DIM);
    let distinct: Vec<Vec<u16>> = (0..8)
        .map(|_| (0..INDEX_DIM).map(|_| bf16_bits(rng.signed())).collect())
        .collect();
    for b in 0..pooled_capacity {
        if case.tie_heavy {
            pooled_rows.extend_from_slice(&distinct[b % 8]);
        } else {
            pooled_rows.extend((0..INDEX_DIM).map(|_| bf16_bits(rng.signed())));
        }
    }
    let bytes = pooled_rows.iter().flat_map(|v| v.to_le_bytes()).collect();
    (query, bytes)
}

/// One rank's buffers for a `(P, n)` tile.
struct Rig {
    position_start: usize,
    rows: usize,
    pooled_capacity: usize,
    query: GpuTensor,
    pooled: GpuTensor,
    sel_ref: GpuTensor,
    mir_ref: GpuTensor,
    sel_sp: GpuTensor,
    mir_sp: GpuTensor,
    compact: GpuTensor,
}

impl Rig {
    fn new(gpu: &mut Gpu, case: &Case) -> Result<Self, String> {
        let pooled_capacity = ((case.position_start + case.rows) / COMPRESS).div_ceil(256) * 256;
        let (query, pooled_bytes) = make_data(case, pooled_capacity);
        let q = gpu
            .upload_f32(&query, &[case.rows * QUERY_STRIDE])
            .map_err(es)?;
        let mut pooled = gpu
            .upload_raw(&pooled_bytes, &[pooled_bytes.len()])
            .map_err(es)?;
        pooled.dtype = DType::BF16;
        pooled.shape = vec![pooled_bytes.len() / 2];
        let junk = |gpu: &mut Gpu, bytes: usize, b: u8| {
            gpu.upload_raw(&vec![b; bytes], &[bytes]).map_err(es)
        };
        let sel_bytes = case.rows * CAPACITY * 4;
        Ok(Self {
            position_start: case.position_start,
            rows: case.rows,
            pooled_capacity,
            query: q,
            pooled,
            sel_ref: junk(gpu, sel_bytes, 0x7B)?,
            mir_ref: junk(gpu, CAPACITY * 4, 0x7B)?,
            sel_sp: junk(gpu, sel_bytes, 0x7B)?,
            mir_sp: junk(gpu, CAPACITY * 4, 0x7B)?,
            compact: junk(gpu, case.rows * BUDGET_BLOCKS * 4, 0x11)?,
        })
    }

    fn free(self, gpu: &mut Gpu) -> Result<(), String> {
        for t in [
            self.query,
            self.pooled,
            self.sel_ref,
            self.mir_ref,
            self.sel_sp,
            self.mir_sp,
            self.compact,
        ] {
            gpu.free_tensor(t).map_err(es)?;
        }
        Ok(())
    }

    /// Select tile rows `[row0, row0 + m)` into `selected` (the tile's full buffer) with the
    /// batched selector; `mirror` receives the call's last row. Returns whether the mirror was
    /// persisted by the kernel.
    fn select(
        &self,
        gpu: &mut Gpu,
        selected: &GpuTensor,
        mirror: Option<&GpuTensor>,
        row0: usize,
        m: usize,
    ) -> Result<bool, String> {
        let q = self.query.sub_offset(row0 * QUERY_STRIDE, m * QUERY_STRIDE);
        let s = selected.sub_offset(row0 * CAPACITY * 4, m * CAPACITY * 4);
        let position_start = self.position_start + row0;
        let p = IndexedAttentionSelectBatch {
            query: &q,
            pooled: &self.pooled,
            pooled_bf16: None,
            pooled_bf16_valid_blocks: 0,
            selected: &s,
            rows: m,
            query_row_stride: QUERY_STRIDE,
            block_count: (position_start + m) / COMPRESS,
            index_heads: INDEX_HEADS,
            index_dim: INDEX_DIM,
            budget_blocks: BUDGET_BLOCKS,
            compress: COMPRESS,
            position_start,
            capacity: CAPACITY,
            shape_blocks: self.pooled_capacity,
        };
        match mirror {
            Some(mir) => indexed_attention_select_batch_mirrored(gpu, &p, mir).map_err(es),
            None => indexed_attention_select_batch(gpu, &p)
                .map(|_| false)
                .map_err(es),
        }
    }

    fn blocks(&self) -> QsaSpBlocks<'_> {
        QsaSpBlocks {
            selected: &self.sel_sp,
            compact: &self.compact,
            tile_rows: self.rows,
            position_start: self.position_start,
            block_count: (self.position_start + self.rows) / COMPRESS,
            budget_blocks: BUDGET_BLOCKS,
            compress: COMPRESS,
            capacity: CAPACITY,
        }
    }

    /// pack own rows, all-gather the compact rows, expand the peer rows.
    fn exchange(
        &self,
        x: &ArHcExchange,
        gpu: &mut Gpu,
        rank: usize,
        own: (usize, usize),
        peer: (usize, usize),
    ) -> Result<(), String> {
        let b = self.blocks();
        gpu.qsa_sp_pack_blocks(&b, own.0, own.1).map_err(es)?;
        // SAFETY: `compact` is a live `[rows][1024 bf16]` buffer on `gpu`'s device and both row
        // ranges lie inside it.
        unsafe {
            x.sp_ag_rows_lockstep(
                rank,
                gpu,
                self.compact.buf.as_ptr(),
                SpAgElem::Bf16,
                2 * BUDGET_BLOCKS,
                own,
                peer,
            )
        }
        .map_err(es)?;
        gpu.qsa_sp_expand_blocks(&b, peer.0, peer.1).map_err(es)
    }
}

/// `row,slot` of the first differing 4-byte word, `None` when equal.
fn first_diff(a: &[u8], b: &[u8]) -> Option<(usize, usize)> {
    if a.len() != b.len() {
        return Some((usize::MAX, usize::MAX));
    }
    a.iter().zip(b).position(|(x, y)| x != y).map(|at| {
        let w = at / 4;
        (w / CAPACITY, w % CAPACITY)
    })
}

fn drain(gpu: &mut Gpu, what: &str) -> Result<(), String> {
    gpu.hip.device_synchronize().map_err(es)?;
    gpu.take_device_wait_error()
        .map_err(|e| format!("{what}: {e}"))
}

/// Own and peer ranges of `rank` for split point `s`.
fn ranges(rank: usize, s: usize, n: usize) -> ((usize, usize), (usize, usize)) {
    let first = (0, s);
    let second = (s, n - s);
    if rank == 0 {
        (first, second)
    } else {
        (second, first)
    }
}

fn run_case(
    gpu: &mut Gpu,
    x: &ArHcExchange,
    bar: &Barrier,
    rank: usize,
    case: &Case,
    lines: &mut Vec<String>,
) -> Result<bool, String> {
    let n = case.rows;
    let s = if case.even_split {
        n.div_ceil(2)
    } else {
        balanced_split(n, case.position_start)
    };
    let (own, peer) = ranges(rank, s, n);
    let rig = Rig::new(gpu, case)?;
    bar.wait();
    // reference: the full-tile mirrored select on this card
    let persisted = rig.select(gpu, &rig.sel_ref, Some(&rig.mir_ref), 0, n)?;
    drain(gpu, "full-tile select")?;
    let ref_sel = gpu.download_raw_bytes(&rig.sel_ref).map_err(es)?;
    let ref_mir = if persisted {
        gpu.download_raw_bytes(&rig.mir_ref).map_err(es)?
    } else {
        ref_sel[(n - 1) * CAPACITY * 4..].to_vec()
    };
    // sequence-parallel path: own rows only, then pack / gather / expand
    let last_rows = own.0 + own.1 == n;
    let sp_persisted = if last_rows {
        rig.select(gpu, &rig.sel_sp, Some(&rig.mir_sp), own.0, own.1)?
    } else {
        rig.select(gpu, &rig.sel_sp, None, own.0, own.1)?
    };
    drain(gpu, "own-range select")?;
    bar.wait();
    rig.exchange(x, gpu, rank, own, peer)?;
    drain(gpu, "pack/gather/expand")?;
    let sp_sel = gpu.download_raw_bytes(&rig.sel_sp).map_err(es)?;
    let sp_mir = if last_rows {
        if sp_persisted {
            gpu.download_raw_bytes(&rig.mir_sp).map_err(es)?
        } else {
            sp_sel[(n - 1) * CAPACITY * 4..].to_vec()
        }
    } else {
        sp_sel[(n - 1) * CAPACITY * 4..].to_vec()
    };
    let sel_diff = first_diff(&sp_sel, &ref_sel);
    let mir_diff = first_diff(&sp_mir, &ref_mir);
    let pass = sel_diff.is_none() && mir_diff.is_none();
    let verdict = |d: Option<(usize, usize)>| match d {
        None => "PASS".to_string(),
        Some((r, c)) => format!("FAIL first_diff={r},{c}"),
    };
    lines.push(format!(
        "rank {rank} {} (P={} n={n} split {s}: own {:?} peer {:?}): selected {}  mirror {}",
        case.name,
        case.position_start,
        own,
        peer,
        verdict(sel_diff),
        match mir_diff {
            None => "PASS".to_string(),
            Some(_) => "FAIL".to_string(),
        }
    ));
    bar.wait();
    rig.free(gpu)?;
    Ok(pass)
}

struct BenchOut {
    position_start: usize,
    rows: usize,
    split: usize,
    full_us: f64,
    own_us: f64,
    exch_us: f64,
}

fn bench_case(
    gpu: &mut Gpu,
    x: &ArHcExchange,
    bar: &Barrier,
    rank: usize,
    position_start: usize,
    n: usize,
) -> Result<BenchOut, String> {
    let case = Case {
        name: "bench",
        position_start,
        rows: n,
        tie_heavy: false,
        even_split: false,
    };
    let s = balanced_split(n, position_start);
    let (own, peer) = ranges(rank, s, n);
    let rig = Rig::new(gpu, &case)?;
    bar.wait();
    let last_rows = own.0 + own.1 == n;
    fn timed<F: FnMut(&mut Gpu) -> Result<(), String>>(
        gpu: &mut Gpu,
        mut f: F,
    ) -> Result<f64, String> {
        gpu.hip.device_synchronize().map_err(es)?;
        let t = Instant::now();
        f(gpu)?;
        gpu.hip.device_synchronize().map_err(es)?;
        Ok(t.elapsed().as_secs_f64() * 1e6)
    }
    let mut full = Vec::new();
    let mut own_t = Vec::new();
    let mut exch = Vec::new();
    for it in 0..BENCH_ITERS + 3 {
        let warm = it < 3;
        let us = timed(gpu, |g| {
            rig.select(g, &rig.sel_ref, Some(&rig.mir_ref), 0, n)
                .map(|_| ())
        })?;
        if !warm {
            full.push(us);
        }
    }
    for it in 0..BENCH_ITERS + 3 {
        let warm = it < 3;
        let us = timed(gpu, |g| {
            let mir = if last_rows { Some(&rig.mir_sp) } else { None };
            rig.select(g, &rig.sel_sp, mir, own.0, own.1).map(|_| ())
        })?;
        if !warm {
            own_t.push(us);
        }
    }
    drain(gpu, "bench select")?;
    for it in 0..BENCH_ITERS + 3 {
        let warm = it < 3;
        bar.wait();
        let us = timed(gpu, |g| rig.exchange(x, g, rank, own, peer))?;
        if !warm {
            exch.push(us);
        }
    }
    drain(gpu, "bench exchange")?;
    bar.wait();
    rig.free(gpu)?;
    Ok(BenchOut {
        position_start,
        rows: n,
        split: s,
        full_us: median_us(full),
        own_us: median_us(own_t),
        exch_us: median_us(exch),
    })
}

struct RankOut {
    lines: Vec<String>,
    pass: bool,
    bench: Vec<BenchOut>,
}

fn run_rank(
    rank: usize,
    device: i32,
    bench: bool,
    x: Arc<ArHcExchange>,
    bar: Arc<Barrier>,
) -> Result<RankOut, String> {
    let mut gpu = Gpu::init_with_device(device).map_err(es)?;
    gpu.ensure_device_wait_block().map_err(es)?;
    x.register(rank, &mut gpu).map_err(es)?;
    bar.wait();
    x.enable_peers(rank, &mut gpu).map_err(es)?;
    bar.wait();
    x.sp_ensure_kernels(&mut gpu).map_err(es)?;
    gpu.ensure_qsa_sp_select_kernels().map_err(es)?;
    x.qualify(rank, &mut gpu, 8, 16)
        .map_err(|e| format!("qualification: {e}"))?;
    bar.wait();
    let mut lines = Vec::new();
    let mut pass = true;
    for case in &CASES {
        pass &= run_case(&mut gpu, &x, &bar, rank, case, &mut lines)?;
    }
    let mut benches = Vec::new();
    if bench {
        for &(p, n) in &BENCH_CASES {
            benches.push(bench_case(&mut gpu, &x, &bar, rank, p, n)?);
        }
    }
    bar.wait();
    x.release(rank, &mut gpu).map_err(es)?;
    Ok(RankOut {
        lines,
        pass,
        bench: benches,
    })
}

fn main() {
    let mut devices = [0i32, 1i32];
    let mut out = ".".to_string();
    let mut bench = false;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--devices" => {
                i += 1;
                let d: Vec<i32> = args[i]
                    .split(',')
                    .map(|s| s.parse().expect("device"))
                    .collect();
                devices = [d[0], d[1]];
            }
            "--out" => {
                i += 1;
                out = args[i].clone();
            }
            "--bench" => bench = true,
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    let x = Arc::new(ArHcExchange::with_mode(SP_ROW_CAP, ArHcMode::Dense).expect("exchange"));
    let bar = Arc::new(Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|rank| {
            let (x, b) = (x.clone(), bar.clone());
            let dev = devices[rank];
            std::thread::spawn(move || match run_rank(rank, dev, bench, x, b) {
                Ok(r) => r,
                Err(e) => {
                    // the peer thread may be parked on a barrier: do not wait for it
                    eprintln!("rank {rank}: FATAL {e}");
                    eprintln!("qsa_sp_gather_oracle: FAIL (rank {rank} fatal)");
                    std::process::exit(3);
                }
            })
        })
        .collect();
    let results: Vec<RankOut> = handles
        .into_iter()
        .map(|h| h.join().expect("thread"))
        .collect();
    let mut report = String::new();
    let mut ok = true;
    for r in &results {
        ok &= r.pass;
        for l in &r.lines {
            report.push_str(l);
            report.push('\n');
        }
    }
    for (rank, r) in results.iter().enumerate() {
        for b in &r.bench {
            report.push_str(&format!(
                "rank {rank} bench P={} n={} split {}: full select {:.1} us  own-range select {:.1} us  pack+AG+expand {:.1} us  expected saving/layer {:.1} us\n",
                b.position_start,
                b.rows,
                b.split,
                b.full_us,
                b.own_us,
                b.exch_us,
                b.full_us - b.own_us - b.exch_us
            ));
        }
    }
    report.push_str(if ok { "ORACLE PASS\n" } else { "ORACLE FAIL\n" });
    print!("{report}");
    let _ = std::fs::create_dir_all(&out);
    let path = format!("{out}/report.txt");
    if let Err(e) = std::fs::write(&path, &report) {
        eprintln!("could not write {path}: {e}");
    }
    std::process::exit(if ok { 0 } else { 1 });
}
