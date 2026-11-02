// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Bitwise oracle + timing for the gfx1151 QSA F32 grouped attention
//! one-denominator-chain twin, against the production incumbent:
//!
//!   incumbent  `indexed_attention_attention_f32_batched_hg12`
//!              module `tensor_ops` (the production source string:
//!              mq_fwht256.h + rope_yarn.h + tensor_ops.hip)
//!   twin       `indexed_attention_attention_f32_batched_hg12_den1`
//!              module `indexed_attention_hg12_den1`
//!              (`kernels/src/indexed_attention_hg12_den1.gfx1151.hip`)
//!
//! Both: block [256,1,1], grid [n_kv_heads, 1, rows], dynamic LDS 0, the same
//! fourteen production arguments (q_with_gate, full_keys, full_values,
//! selected_indices, output, rows, position_start, n_heads, n_kv_heads,
//! head_dim, budget_blocks, compress, capacity, full_capacity) laid out as
//! `indexed_attention_attention_batch_arena` (tensor_ops.rs) builds them:
//! five 8-byte pointers, nine i32, `pad_to(16)`.
//!
//! The variants live in the `VARIANTS` table (index 0 = incumbent = the bitwise
//! reference). Sources are compiled straight from the `.hip` files
//! (`include_str!`); no dispatch/production code is touched. A short SHA-256
//! prefix of each source string is printed at startup.
//!
//! ## What is verified
//! Every f32 bit of `output` of the twin (the whole `rows * n_heads * 256`
//! region PLUS a 256-float guard band after it) equals the incumbent's. Before
//! each variant's correctness launch the buffer is refilled with the NaN-payload
//! sentinel `0x7fc01234`. Required: every cell of the main region is written
//! (no sentinel left), the guard band is untouched, no cell is non-finite, and
//! the incumbent has at least one finite nonzero cell.
//!
//! ## Shapes (Qwen4 production)
//! n_heads = 24, n_kv_heads = 2 (hg12), head_dim = 256, compress = 4,
//! budget_blocks = 512, capacity = 2051, full_capacity = 36864.
//! `q_with_gate` [rows, 2*24*256] (head h: query at h*512, gate at h*512+256),
//! `full_keys`/`full_values` [full_capacity, 2*256], `selected_indices`
//! [rows, capacity] i32, `output` [rows, 24*256]. q/K/V are seeded uniform in
//! [-1, 1) (K/V built once per process, shared by every case).
//!
//! ## Selection (production QSA shape, not a CPU imitation of the selector)
//! Per row: visible = position_start + row + 1; block_count = visible/compress;
//! chosen = min(budget_blocks, block_count); selected_len =
//! min(chosen*compress + visible - block_count*compress, capacity). The first
//! selected_len entries are `chosen` distinct random blocks (random block order,
//! each contributing its `compress` consecutive tokens) followed by the tail
//! tokens block_count*compress .. visible-1; the rest up to capacity is -1.
//!
//! ## Cases (default run)
//!   r{R}           R in 16, 33, 145, 273, 401 at position_start 32493
//!                  (selected_len 2048..2051)
//!   short100_r16   position_start 100, rows 16 (selected_len < 256)
//!   short1000_r33  position_start 1000, rows 33 (selected_len not a multiple of 8)
//!   invalid33      rows 33 at 32493 with injected invalid entries: -1, tokens
//!                  >= visible, tokens >= full_capacity, i32::MAX, negative
//!                  garbage, duplicates, an all-invalid first tile (row 8), an
//!                  all-invalid row (row 7: zero denominator), invalid tail entry.
//! `--rows=a,b,c` replaces the five r{R} cases (no short / invalid cases).
//!
//! ## Timing
//! Per variant WARMUP launches, then SAMPLES individually HIP-event-timed
//! launches, launch order cyclically shifted per sample. Reported: median and
//! min per launch and incumbent_median / variant_median.
//!
//! GFX1151 ONLY: refuses any other arch (exit 2) unless `--precheck` admits
//! gfx1100 / gfx1201 for a correctness-only pre-check (timings then are NOT
//! gfx1151 numbers). Exit 1 on any mismatch.
//!
//! Run:
//!   cargo run --release -p rdna-compute --features lab --example test_qsa_hg12_twin_gfx1151 [-- --rows=145,273 --precheck] [case-name-filter]

use hip_bridge::KernargBlob;
use rdna_compute::{Gpu, GpuTensor};
use sha2::{Digest, Sha256};

/// Byte-identical to `TENSOR_OPS_SRC` in `crates/rdna-compute/src/tensor_ops.rs`.
const SRC_INCUMBENT: &str = concat!(
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/rope_yarn.h"),
    include_str!("../../../kernels/src/tensor_ops.hip")
);
const SRC_TWIN: &str = include_str!("../../../kernels/src/indexed_attention_hg12_den1.gfx1151.hip");
const MODULE_INCUMBENT: &str = "tensor_ops";
const MODULE_TWIN: &str = "indexed_attention_hg12_den1";

#[derive(Clone, Copy)]
struct Variant {
    label: &'static str,
    module: &'static str,
    func: &'static str,
    src: &'static str,
}

/// Index 0 is the incumbent (the bitwise reference); the rest are twins.
const VARIANTS: [Variant; 2] = [
    Variant {
        label: "incumbent",
        module: MODULE_INCUMBENT,
        func: "indexed_attention_attention_f32_batched_hg12",
        src: SRC_INCUMBENT,
    },
    Variant {
        label: "den1",
        module: MODULE_TWIN,
        func: "indexed_attention_attention_f32_batched_hg12_den1",
        src: SRC_TWIN,
    },
];

const N_HEADS: usize = 24;
const N_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const COMPRESS: usize = 4;
const BUDGET_BLOCKS: usize = 512;
const CAPACITY: usize = 2051;
const FULL_CAPACITY: usize = 36864;
const KV_STRIDE: usize = N_KV_HEADS * HEAD_DIM;
const Q_ROW: usize = 2 * N_HEADS * HEAD_DIM;
const OUT_ROW: usize = N_HEADS * HEAD_DIM;
/// hg12 shape bound (QSA_ATTENTION_HG12_MAX_SELECTED = 9 * 256).
const HG12_MAX_SELECTED: usize = 9 * 256;
const _: () = assert!(CAPACITY <= HG12_MAX_SELECTED);
const LONG_POS: usize = 32493;
const SEED: u64 = 0x5153_4131_3248_4731; // "QSA12HG1"
const WARMUP: usize = 3;
const SAMPLES: usize = 15;
const GUARD_FLOATS: usize = 256;
const SENTINEL_BITS: u32 = 0x7fc0_1234;

const DEFAULT_ROWS: [usize; 5] = [16, 33, 145, 273, 401];

// ── RNG ────────────────────────────────────────────────────────────────────

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() >> 11) as u128 * n as u128 >> 53) as usize
    }
    /// Uniform f32 in [-1, 1) with 24 random mantissa bits.
    fn sym(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

// ── cases ──────────────────────────────────────────────────────────────────

struct Spec {
    name: String,
    rows: usize,
    position_start: usize,
    /// Inject invalid selected entries (kernel must treat them as dead rows).
    invalid: bool,
    /// Long-context production case (counts toward the geometric mean).
    long: bool,
    /// Seed salt: a case is seeded by (kind, rows, position), never by list position.
    salt: u64,
}

fn long_case(rows: usize) -> Spec {
    Spec {
        name: format!("r{rows}"),
        rows,
        position_start: LONG_POS,
        invalid: false,
        long: true,
        salt: 0x11,
    }
}

fn specs(rows_override: Option<&[usize]>) -> Vec<Spec> {
    if let Some(rows) = rows_override {
        return rows.iter().map(|&r| long_case(r)).collect();
    }
    let mut v: Vec<Spec> = DEFAULT_ROWS.iter().map(|&r| long_case(r)).collect();
    v.push(Spec {
        name: "short100_r16".to_string(),
        rows: 16,
        position_start: 100,
        invalid: false,
        long: false,
        salt: 0x22,
    });
    v.push(Spec {
        name: "short1000_r33".to_string(),
        rows: 33,
        position_start: 1000,
        invalid: false,
        long: false,
        salt: 0x33,
    });
    v.push(Spec {
        name: "invalid33".to_string(),
        rows: 33,
        position_start: LONG_POS,
        invalid: true,
        long: false,
        salt: 0x44,
    });
    v
}

// ── host helpers ───────────────────────────────────────────────────────────

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding/invalid byte patterns; length in bytes is exact.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn sha_prefix(s: &str) -> String {
    Sha256::digest(s.as_bytes())
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The kernel's own selected_len formula.
fn selected_len(position_start: usize, row: usize) -> usize {
    let visible = position_start + row + 1;
    let block_count = visible / COMPRESS;
    let chosen = BUDGET_BLOCKS.min(block_count);
    (chosen * COMPRESS + visible - block_count * COMPRESS).min(CAPACITY)
}

/// Production-shaped selection: `chosen` distinct random blocks in random order
/// (each `COMPRESS` consecutive tokens), then the tail tokens; rest -1.
/// Returns the flat [rows, CAPACITY] table.
fn build_selected(rng: &mut Rng, rows: usize, position_start: usize) -> Vec<i32> {
    let mut out = vec![-1i32; rows * CAPACITY];
    let mut perm: Vec<u32> = Vec::new();
    for row in 0..rows {
        let visible = position_start + row + 1;
        let block_count = visible / COMPRESS;
        let chosen = BUDGET_BLOCKS.min(block_count);
        let len = selected_len(position_start, row);
        perm.clear();
        perm.extend(0..block_count as u32);
        let dst = &mut out[row * CAPACITY..(row + 1) * CAPACITY];
        let mut n = 0usize;
        for j in 0..chosen {
            let r = j + rng.below(block_count - j);
            perm.swap(j, r);
            for c in 0..COMPRESS {
                dst[n] = (perm[j] as usize * COMPRESS + c) as i32;
                n += 1;
            }
        }
        for tok in block_count * COMPRESS..visible {
            dst[n] = tok as i32;
            n += 1;
        }
        // Entries past selected_len (capacity clamp) stay -1.
        for e in dst[len.min(n)..].iter_mut() {
            *e = -1;
        }
    }
    out
}

/// Inject entries the kernel must treat as invalid rows. Returns a short
/// description of what was injected (for the log).
fn inject_invalid(sel: &mut [i32], rows: usize, position_start: usize) -> String {
    let mut counts = [0usize; 8];
    for row in 0..rows {
        let visible = (position_start + row + 1) as i32;
        let len = selected_len(position_start, row);
        let dst = &mut sel[row * CAPACITY..(row + 1) * CAPACITY];
        if row == 7 {
            // Whole row invalid: zero denominator, output 0.
            for e in dst[..len].iter_mut() {
                *e = -1;
            }
            counts[0] += 1;
            continue;
        }
        if row == 8 {
            // First 256-entry tile entirely invalid, later tiles live.
            for e in dst[..256.min(len)].iter_mut() {
                *e = -1;
            }
            counts[1] += 1;
        }
        if row % 4 == 0 && len > 4 {
            dst[3] = -1;
            counts[2] += 1;
        }
        if row % 3 == 1 && len > 130 {
            dst[100] = visible + 5; // >= visible, < full_capacity
            dst[101] = visible; // == visible
            counts[3] += 1;
        }
        if row % 5 == 2 && len > 300 {
            dst[257] = FULL_CAPACITY as i32 + 10; // >= full_capacity
            dst[258] = FULL_CAPACITY as i32;
            dst[259] = i32::MAX;
            dst[260] = -7;
            counts[4] += 1;
        }
        if row % 7 == 3 && len > 1 {
            dst[len - 1] = -1; // invalid last entry (partial tile)
            counts[5] += 1;
        }
        if row % 6 == 5 && len > 13 {
            dst[12] = dst[11]; // duplicate token (a valid repeat, still in-order sums)
            counts[6] += 1;
        }
    }
    format!(
        "all-invalid rows={} dead-first-tile rows={} -1 entries={} >=visible rows={} \
         >=full_capacity/i32::MAX/neg rows={} dead-tail rows={} duplicate rows={}",
        counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6]
    )
}

// ── shared K/V ─────────────────────────────────────────────────────────────

struct Kv {
    keys: GpuTensor,
    values: GpuTensor,
}

impl Kv {
    fn build(gpu: &mut Gpu, seed: u64) -> Kv {
        let n = FULL_CAPACITY * KV_STRIDE;
        let mut rng = Rng(seed ^ 0x4B45_5953);
        let k: Vec<f32> = (0..n).map(|_| rng.sym()).collect();
        let keys = gpu.upload_f32(&k, &[n]).expect("keys upload");
        drop(k);
        let mut rng = Rng(seed ^ 0x5641_4C53);
        let v: Vec<f32> = (0..n).map(|_| rng.sym()).collect();
        let values = gpu.upload_f32(&v, &[n]).expect("values upload");
        Kv { keys, values }
    }
    fn free(self, gpu: &mut Gpu) {
        gpu.free_tensor(self.keys).expect("free keys");
        gpu.free_tensor(self.values).expect("free values");
    }
}

// ── one case ───────────────────────────────────────────────────────────────

struct CaseOutcome {
    name: String,
    rows: usize,
    position_start: usize,
    sel_min: usize,
    sel_max: usize,
    long: bool,
    ok: bool,
    /// Indexed by VARIANTS index.
    median_us: Vec<f64>,
}

fn run_case(gpu: &mut Gpu, spec: &Spec, kv: &Kv) -> CaseOutcome {
    let rows = spec.rows;
    let mut rng = Rng(SEED
        ^ ((rows as u64) << 32)
        ^ ((spec.position_start as u64) << 8)
        ^ (spec.salt << 4)
        ^ 0xA5A5);
    assert!(
        spec.position_start + rows <= FULL_CAPACITY,
        "case exceeds full_capacity"
    );
    let lens: Vec<usize> = (0..rows)
        .map(|r| selected_len(spec.position_start, r))
        .collect();
    let sel_min = *lens.iter().min().unwrap();
    let sel_max = *lens.iter().max().unwrap();
    assert!(sel_max <= CAPACITY, "selected_len above hg12 bound");

    let mut selected = build_selected(&mut rng, rows, spec.position_start);
    let inject = spec
        .invalid
        .then(|| inject_invalid(&mut selected, rows, spec.position_start));

    let q: Vec<f32> = (0..rows * Q_ROW).map(|_| rng.sym()).collect();
    let q_t = gpu.upload_f32(&q, &[q.len()]).expect("q upload");
    let sel_t = gpu
        .upload_raw(&i32_bytes(&selected), &[selected.len() * 4])
        .expect("selected");

    // Output with a guard band; every variant starts from the NaN-payload sentinel.
    let main_cells = rows * OUT_ROW;
    let y_floats = main_cells + GUARD_FLOATS;
    let sentinel = vec![f32::from_bits(SENTINEL_BITS); y_floats];
    let y_t = gpu.upload_f32(&sentinel, &[y_floats]).expect("y upload");

    let live_entries: usize = selected
        .chunks_exact(CAPACITY)
        .enumerate()
        .map(|(r, row)| {
            let visible = (spec.position_start + r + 1) as i64;
            row[..lens[r]]
                .iter()
                .filter(|&&t| t >= 0 && (t as i64) < visible)
                .count()
        })
        .sum();
    println!(
        "\n=== {} ===\n  shape n_heads={N_HEADS} n_kv_heads={N_KV_HEADS} head_dim={HEAD_DIM} \
         compress={COMPRESS} budget_blocks={BUDGET_BLOCKS} capacity={CAPACITY} \
         full_capacity={FULL_CAPACITY}\n  rows={rows} position_start={} (visible {}..{}) \
         selected_len {sel_min}..{sel_max} live entries={live_entries} output {:.2} MB",
        spec.name,
        spec.position_start,
        spec.position_start + 1,
        spec.position_start + rows,
        (main_cells * 4) as f64 / 1e6,
    );
    if let Some(desc) = &inject {
        println!("  injected: {desc}");
    }

    // Kernarg blobs: mirrors indexed_attention_attention_batch_arena
    // (tensor_ops.rs: five pointers, rows, position_start, then
    // n_heads, n_kv_heads, head_dim, budget_blocks, compress, capacity,
    // full_capacity, pad_to(16)).
    let blob = || -> Vec<u8> {
        let mut b = KernargBlob::new();
        for t in [&q_t, &kv.keys, &kv.values, &sel_t, &y_t] {
            b.push_ptr(t.buf.as_ptr());
        }
        b.push_i32(rows as i32);
        b.push_i32(spec.position_start as i32);
        for v in [
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            BUDGET_BLOCKS,
            COMPRESS,
            CAPACITY,
            FULL_CAPACITY,
        ] {
            b.push_i32(v as i32);
        }
        b.pad_to(16);
        b.into_vec()
    };
    let mut blobs: Vec<Vec<u8>> = VARIANTS.iter().map(|_| blob()).collect();
    let grid = [N_KV_HEADS as u32, 1u32, rows as u32];
    let block = [256u32, 1u32, 1u32];

    // Correctness: one fresh-sentinel run per variant, compared against the
    // incumbent's output (the first one run).
    let mut ok = true;
    let mut reference: Vec<f32> = Vec::new();
    for (vi, v) in VARIANTS.iter().enumerate() {
        gpu.hip
            .memcpy_htod(&y_t.buf, f32_bytes(&sentinel))
            .expect("reset y");
        gpu.launch_kernel_blob(v.func, grid, block, 0, &mut blobs[vi])
            .expect("launch");
        gpu.hip.device_synchronize().expect("sync");
        let out = gpu.download_f32(&y_t).expect("download y");
        let unwritten = out[..main_cells]
            .iter()
            .filter(|x| x.to_bits() == SENTINEL_BITS)
            .count();
        let guard_bad = out[main_cells..]
            .iter()
            .filter(|x| x.to_bits() != SENTINEL_BITS)
            .count();
        let nonfinite = out[..main_cells]
            .iter()
            .filter(|x| !x.is_finite() && x.to_bits() != SENTINEL_BITS)
            .count();
        let nonzero = out[..main_cells]
            .iter()
            .filter(|x| x.is_finite() && x.to_bits() << 1 != 0)
            .count();
        let mut line = format!(
            "  [{:<9}] unwritten={unwritten} guard_clobbered={guard_bad} nonfinite={nonfinite} \
             nonzero_cells={nonzero}",
            v.label
        );
        let mut v_ok = unwritten == 0 && guard_bad == 0 && nonfinite == 0;
        if vi == 0 {
            if live_entries > 0 && nonzero == 0 {
                v_ok = false;
                line.push_str(" ALL-ZERO");
            }
            line.push_str(if v_ok {
                " (reference sane)"
            } else {
                " REFERENCE-BAD"
            });
            reference = out;
        } else {
            let mut count = 0usize;
            let mut first = None;
            for j in 0..out.len() {
                if out[j].to_bits() != reference[j].to_bits() {
                    count += 1;
                    first.get_or_insert(j);
                }
            }
            match first {
                None => line.push_str(" | BITWISE-IDENTICAL to incumbent"),
                Some(j) => {
                    v_ok = false;
                    line.push_str(&format!(
                        " | MISMATCH {count} cells; first at row {} head {} ch {}: {:#010x} vs \
                         incumbent {:#010x}",
                        j / OUT_ROW,
                        (j % OUT_ROW) / HEAD_DIM,
                        j % HEAD_DIM,
                        out[j].to_bits(),
                        reference[j].to_bits()
                    ));
                }
            }
            if !v_ok && unwritten + guard_bad + nonfinite > 0 {
                line.push_str(" | (unwritten/guard/nonfinite violation)");
            }
        }
        ok &= v_ok;
        println!("{line}");
    }
    drop(reference);

    // Timing: individually event-timed launches, order cyclically shifted per sample.
    let ev0 = gpu.hip.event_create().expect("event");
    let ev1 = gpu.hip.event_create().expect("event");
    let time_one = |gpu: &mut Gpu, vi: usize, blobs: &mut Vec<Vec<u8>>| -> f64 {
        let v = &VARIANTS[vi];
        gpu.hip.event_record(&ev0, None).expect("rec");
        gpu.launch_kernel_blob(v.func, grid, block, 0, &mut blobs[vi])
            .expect("launch");
        gpu.hip.event_record(&ev1, None).expect("rec");
        gpu.hip.event_synchronize(&ev1).expect("evsync");
        gpu.hip.event_elapsed_ms(&ev0, &ev1).expect("elapsed") as f64 * 1000.0
    };
    let n = VARIANTS.len();
    for s in 0..WARMUP {
        for j in 0..n {
            time_one(gpu, (s + j) % n, &mut blobs);
        }
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); n];
    for s in 0..SAMPLES {
        for j in 0..n {
            let vi = (s + j) % n;
            let t = time_one(gpu, vi, &mut blobs);
            samples[vi].push(t);
        }
    }
    gpu.hip.event_destroy(ev0).expect("destroy");
    gpu.hip.event_destroy(ev1).expect("destroy");

    let mut median_us = vec![f64::NAN; n];
    println!(
        "  timing: {SAMPLES} individual launches/variant after {WARMUP} warmup, order cyclically \
         shifted per sample"
    );
    for vi in 0..n {
        let min = samples[vi].iter().cloned().fold(f64::INFINITY, f64::min);
        let med = median(&mut samples[vi]);
        median_us[vi] = med;
        println!(
            "  {:<9} median {:>9.2} us  min {:>9.2} us  ({:.3}x incumbent median)",
            VARIANTS[vi].label,
            med,
            min,
            median_us[0] / med
        );
    }
    println!(
        "  VERDICT {}: {}",
        spec.name,
        if ok { "PASS (bitwise)" } else { "FAIL" }
    );

    for t in [q_t, sel_t, y_t] {
        gpu.free_tensor(t).expect("free");
    }
    CaseOutcome {
        name: spec.name.clone(),
        rows,
        position_start: spec.position_start,
        sel_min,
        sel_max,
        long: spec.long,
        ok,
        median_us,
    }
}

fn main() {
    let mut rows_override: Option<Vec<usize>> = None;
    let mut filter: Option<String> = None;
    // `--precheck`: correctness pre-check on a gfx1100 / gfx1201 card: bitwise
    // verdicts are meaningful, timings are not gfx1151 numbers.
    let mut precheck = false;
    for a in std::env::args().skip(1) {
        if let Some(list) = a.strip_prefix("--rows=") {
            let v: Vec<usize> = list
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse()
                        .unwrap_or_else(|_| panic!("bad --rows entry {s:?}"))
                })
                .collect();
            assert!(
                !v.is_empty() && v.iter().all(|&r| r > 0),
                "--rows needs positive counts"
            );
            rows_override = Some(v);
        } else if a == "--precheck" {
            precheck = true;
        } else if a.starts_with("--") {
            eprintln!("unknown flag {a} (supported: --rows=a,b,c, --precheck)");
            std::process::exit(2);
        } else {
            filter = Some(a);
        }
    }

    let mut gpu = Gpu::init().expect("GPU init");
    println!("arch = {}", gpu.arch);
    let gfx1151 = gpu.arch_caps.is_gfx1151() && gpu.arch.starts_with("gfx1151");
    if !gfx1151
        && !(precheck && (gpu.arch.starts_with("gfx1100") || gpu.arch.starts_with("gfx1201")))
    {
        eprintln!(
            "REFUSING: test_qsa_hg12_twin_gfx1151 is a gfx1151-only oracle (found {}; \
             --precheck admits a gfx1100/gfx1201 correctness pre-check).",
            gpu.arch
        );
        std::process::exit(2);
    }
    if !gfx1151 {
        println!(
            "NOTE: {} correctness pre-check; timings below are NOT gfx1151 numbers",
            gpu.arch
        );
    }

    println!(
        "source sha256[..8]: incumbent {} ({MODULE_INCUMBENT}: mq_fwht256.h+rope_yarn.h+tensor_ops.hip), \
         twin {} (indexed_attention_hg12_den1.gfx1151.hip)",
        sha_prefix(SRC_INCUMBENT),
        sha_prefix(SRC_TWIN)
    );
    for v in &VARIANTS {
        gpu.ensure_kernel_public(v.module, v.src, v.func)
            .unwrap_or_else(|e| panic!("compile {}: {e:?}", v.func));
        println!(
            "  loaded {:<9} {} (block 256, grid [{N_KV_HEADS},1,rows])",
            v.label, v.func
        );
    }

    let all_specs = specs(rows_override.as_deref());
    let mut kv: Option<Kv> = None;
    let mut outcomes = Vec::new();
    for sp in all_specs.iter() {
        if filter.as_deref().is_some_and(|f| !sp.name.contains(f)) {
            continue;
        }
        if kv.is_none() {
            println!(
                "\n[building shared K/V: {FULL_CAPACITY} x {KV_STRIDE} f32 each, {:.2} MB each]",
                (FULL_CAPACITY * KV_STRIDE * 4) as f64 / 1e6
            );
            kv = Some(Kv::build(&mut gpu, SEED));
        }
        outcomes.push(run_case(&mut gpu, sp, kv.as_ref().unwrap()));
    }
    if let Some(old) = kv.take() {
        old.free(&mut gpu);
    }

    let mut all_ok = !outcomes.is_empty();
    println!("\n================ SUMMARY (median us; ratio = incumbent median / variant median) ================");
    print!(
        "{:<14} {:>5} {:>7} {:>11} {:>7} | {:>9}",
        "case", "rows", "pos", "sel_len", "verdict", "inc us"
    );
    for v in VARIANTS.iter().skip(1) {
        print!(" | {:>9} {:>6}", format!("{} us", v.label), "x");
    }
    println!();
    for o in &outcomes {
        all_ok &= o.ok;
        print!(
            "{:<14} {:>5} {:>7} {:>11} {:>7} | {:>9.2}",
            o.name,
            o.rows,
            o.position_start,
            format!("{}..{}", o.sel_min, o.sel_max),
            if o.ok { "PASS" } else { "FAIL" },
            o.median_us[0]
        );
        for i in 1..VARIANTS.len() {
            print!(
                " | {:>9.2} {:>6.3}",
                o.median_us[i],
                o.median_us[0] / o.median_us[i]
            );
        }
        println!();
    }

    let geo: Vec<&CaseOutcome> = outcomes.iter().filter(|o| o.long).collect();
    if geo.is_empty() {
        println!("\ngeomean ratio: no long-context cases ran");
    } else {
        println!(
            "\ngeometric-mean ratio (incumbent/variant median) over {} long-context cases:",
            geo.len()
        );
        for i in 1..VARIANTS.len() {
            let ln: f64 = geo
                .iter()
                .map(|o| (o.median_us[0] / o.median_us[i]).ln())
                .sum::<f64>()
                / geo.len() as f64;
            println!("  {:<9} {:.4}x", VARIANTS[i].label, ln.exp());
        }
    }
    println!(
        "OVERALL: {}",
        if all_ok {
            "PASS (all bitwise identical)"
        } else {
            "FAIL"
        }
    );
    std::process::exit(if all_ok { 0 } else { 1 });
}
