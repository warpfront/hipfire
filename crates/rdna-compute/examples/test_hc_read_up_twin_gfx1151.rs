// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Bitwise oracle + timing for the gfx1151 HC read tail (BF16 up projection
//! fused with hyper_read_projected_f32) scheduling twins, against the
//! production incumbent:
//!
//!   incumbent  `hyper_read_up_fused_f32`, module `tensor_ops`
//!              (source = mq_fwht256.h + rope_yarn.h + tensor_ops.hip, i.e.
//!              `TENSOR_OPS_SRC`), grid [hidden/8, ceil(rows/128), 1],
//!              block [256,1,1], dynamic LDS 32 * (K/2 + 1) * 4 bytes
//!   twin       `hyper_read_up_fused_f32_sx`, module `hyper_read_up_fused_twin`
//!              (`kernels/src/hyper_read_up_fused_twin.gfx1151.hip`), same
//!              grid/block/LDS and the same eight kernel arguments.
//!
//! The variants live in the `VARIANTS` table: adding/removing a twin is one
//! line. Kernels are compiled straight from the `.hip` sources
//! (`include_str!`); no dispatch/production code is touched. A short SHA-256
//! prefix of each source string is printed at startup.
//!
//! ## Kernel arguments (mirrors `hyper_read_up_fused`, tensor_ops.rs)
//! W (BF16 bits u16, [4*hidden, K] row-major), X (f32 [rows, K]), normalized
//! (f32 [rows, 4*hidden], or BF16 bits u16 when normalized_bf16 = 1), mixed
//! (f32 [rows, hidden], output), i32 hidden, i32 K, i32 N (= rows), i32
//! normalized_bf16, `pad_to(16)`.
//!
//! ## What is verified
//! Every f32 bit of `mixed` of each twin (the whole `rows * hidden` region PLUS
//! a 256-float guard band after it) equals the incumbent's. Before each
//! variant's correctness launch the buffer is refilled with the NaN-payload
//! sentinel `0x7fc01234`. Required: the guard band is untouched, no cell is
//! non-finite or left at the sentinel in the incumbent's output, and the
//! incumbent has at least one finite nonzero cell.
//!
//! ## Data
//! hidden = 2560, K (low_rank) = 320 (Qwen4 production). W: BF16 bits of
//! finite values with |w| in ~[1e-3, 0.1] (random sign, round-to-nearest-even
//! truncation of an f32). X: f32 uniform [-1, 1). normalized: uniform [-2, 2)
//! (f32, or its BF16 bits for the bf16 cases).
//!
//! ## Cases (default run)
//!   r{R}    rows R in 16, 33, 145, 273, 401, 529, normalized f32
//!   r{R}b   rows 145, 401, normalized BF16
//!   r1, r131  partial passes (rows not a multiple of the 128-token block /
//!             the 8-wave pass), normalized f32
//! `--rows=a,b,c` replaces the list (f32 cases; an entry with a `b` suffix,
//! e.g. `145b`, runs the BF16-normalized variant).
//!
//! ## Timing
//! Per variant WARMUP launches, then SAMPLES individually HIP-event-timed
//! launches, launch order cyclically shifted per sample. Reported: median and
//! min per launch, incumbent_median / variant_median.
//!
//! GFX1151 ONLY: refuses any other arch (exit 2); `--precheck` admits gfx1100 /
//! gfx1201 for a correctness pre-check (timings are then not gfx1151 numbers).
//! Exit 1 on any mismatch.
//!
//! Run:
//!   cargo run --release -p rdna-compute --features lab --example test_hc_read_up_twin_gfx1151 [-- --rows=145,273 --variants=sx --precheck] [case-name-filter]

use hip_bridge::KernargBlob;
use rdna_compute::Gpu;
use sha2::{Digest, Sha256};

const SRC_INCUMBENT: &str = concat!(
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/rope_yarn.h"),
    include_str!("../../../kernels/src/tensor_ops.hip")
);
const SRC_TWIN: &str = include_str!("../../../kernels/src/hyper_read_up_fused_twin.gfx1151.hip");
const MODULE_INCUMBENT: &str = "tensor_ops";
const MODULE_TWIN: &str = "hyper_read_up_fused_twin";

#[derive(Clone, Copy)]
struct Variant {
    label: &'static str,
    module: &'static str,
    func: &'static str,
    src: &'static str,
}

const fn twin(label: &'static str, func: &'static str) -> Variant {
    Variant {
        label,
        module: MODULE_TWIN,
        func,
        src: SRC_TWIN,
    }
}

/// Index 0 is the incumbent (the bitwise reference); the rest are twins.
const VARIANTS: [Variant; 2] = [
    Variant {
        label: "incumbent",
        module: MODULE_INCUMBENT,
        func: "hyper_read_up_fused_f32",
        src: SRC_INCUMBENT,
    },
    twin("sx", "hyper_read_up_fused_f32_sx"),
];

const HIDDEN: usize = 2560;
const LOW_RANK: usize = 320; // K
const WIDE: usize = 4 * HIDDEN;
const SEED: u64 = 0x4843_5550_5457_494e; // "HCUPTWIN"
const WARMUP: usize = 3;
const SAMPLES: usize = 15;
const GUARD_FLOATS: usize = 256;
const SENTINEL_BITS: u32 = 0x7fc0_1234;
const LDS_BYTES: u32 = (32 * (LOW_RANK / 2 + 1) * 4) as u32;

const MAIN_ROWS: [usize; 6] = [16, 33, 145, 273, 401, 529];
const BF16_ROWS: [usize; 2] = [145, 401];
const PARTIAL_ROWS: [usize; 2] = [1, 131];

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
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Round-to-nearest-even f32 -> BF16 bits (inputs here are always finite).
fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
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

// ── cases ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Spec {
    name: String,
    rows: usize,
    normalized_bf16: bool,
    /// Counted in the per-twin geometric-mean ratio.
    geo: bool,
}

fn spec(rows: usize, normalized_bf16: bool, geo: bool) -> Spec {
    Spec {
        name: format!("r{rows}{}", if normalized_bf16 { "b" } else { "" }),
        rows,
        normalized_bf16,
        geo,
    }
}

fn specs(rows_override: Option<&[(usize, bool)]>) -> Vec<Spec> {
    if let Some(list) = rows_override {
        return list.iter().map(|&(r, b)| spec(r, b, !b)).collect();
    }
    let mut v: Vec<Spec> = MAIN_ROWS.iter().map(|&r| spec(r, false, true)).collect();
    v.extend(BF16_ROWS.iter().map(|&r| spec(r, true, false)));
    v.extend(PARTIAL_ROWS.iter().map(|&r| spec(r, false, false)));
    v
}

// ── one case ───────────────────────────────────────────────────────────────

struct CaseOutcome {
    name: String,
    rows: usize,
    geo: bool,
    ok: bool,
    /// Indexed by VARIANTS index; NaN for variants that were not run.
    median_us: Vec<f64>,
}

fn run_case(
    gpu: &mut Gpu,
    spec: &Spec,
    w_t: &rdna_compute::GpuTensor,
    active: &[usize],
) -> CaseOutcome {
    assert_eq!(
        active.first(),
        Some(&0),
        "incumbent must be the reference variant"
    );
    let rows = spec.rows;
    let mut rng = Rng(SEED ^ ((rows as u64) << 32) ^ ((spec.normalized_bf16 as u64) << 8) ^ 0xA5A5);

    // X [rows, K] uniform [-1, 1); normalized [rows, 4*hidden] uniform [-2, 2).
    let xs: Vec<f32> = (0..rows * LOW_RANK)
        .map(|_| (rng.unit() * 2.0 - 1.0) as f32)
        .collect();
    let norm: Vec<f32> = (0..rows * WIDE)
        .map(|_| (rng.unit() * 4.0 - 2.0) as f32)
        .collect();
    let x_t = gpu.upload_f32(&xs, &[xs.len()]).expect("x upload");
    let n_t = if spec.normalized_bf16 {
        let bits: Vec<u16> = norm.iter().map(|&v| bf16_bits(v)).collect();
        gpu.upload_raw(&u16_bytes(&bits), &[bits.len() * 2])
            .expect("normalized bf16 upload")
    } else {
        gpu.upload_f32(&norm, &[norm.len()])
            .expect("normalized upload")
    };
    drop(norm);

    // mixed with a guard band; every variant starts from the NaN-payload sentinel.
    let main_cells = rows * HIDDEN;
    let y_floats = main_cells + GUARD_FLOATS;
    let sentinel = vec![f32::from_bits(SENTINEL_BITS); y_floats];
    let y_t = gpu.upload_f32(&sentinel, &[y_floats]).expect("y upload");

    let grid = [(HIDDEN / 8) as u32, rows.div_ceil(128) as u32, 1u32];
    let block = [256u32, 1, 1];
    println!(
        "\n=== {} ===\n  shape hidden={HIDDEN} K={LOW_RANK} rows={rows} normalized={} grid=[{}, {}, 1] \
         block=[256,1,1] lds={LDS_BYTES} B",
        spec.name,
        if spec.normalized_bf16 { "bf16" } else { "f32" },
        grid[0],
        grid[1],
    );

    // Kernarg blobs: exactly the production launcher's layout.
    let blob = || -> Vec<u8> {
        let mut b = KernargBlob::new();
        b.push_ptr(w_t.buf.as_ptr());
        b.push_ptr(x_t.buf.as_ptr());
        b.push_ptr(n_t.buf.as_ptr());
        b.push_ptr(y_t.buf.as_ptr());
        b.push_i32(HIDDEN as i32);
        b.push_i32(LOW_RANK as i32);
        b.push_i32(rows as i32);
        b.push_i32(i32::from(spec.normalized_bf16));
        b.pad_to(16);
        b.into_vec()
    };
    let mut blobs: Vec<Vec<u8>> = VARIANTS.iter().map(|_| blob()).collect();

    // Correctness: one fresh-sentinel run per variant, compared against the
    // incumbent's output (the first one run).
    let mut ok = true;
    let mut reference: Vec<f32> = Vec::new();
    for (pos, &vi) in active.iter().enumerate() {
        let v = &VARIANTS[vi];
        gpu.hip
            .memcpy_htod(&y_t.buf, f32_bytes(&sentinel))
            .expect("reset y");
        gpu.launch_kernel_blob(v.func, grid, block, LDS_BYTES, &mut blobs[vi])
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
        let mut v_ok = guard_bad == 0 && nonfinite == 0;
        if pos == 0 {
            if nonzero == 0 {
                v_ok = false;
                line.push_str(" ALL-ZERO");
            }
            if unwritten != 0 {
                v_ok = false;
                line.push_str(" UNWRITTEN-CELLS");
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
                        " | MISMATCH {count} cells; first at row {} col {}: {:#010x} vs \
                         incumbent {:#010x}",
                        j / HIDDEN,
                        j % HIDDEN,
                        out[j].to_bits(),
                        reference[j].to_bits()
                    ));
                }
            }
            if !v_ok && guard_bad + nonfinite > 0 {
                line.push_str(" | (guard/nonfinite violation)");
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
        gpu.launch_kernel_blob(v.func, grid, block, LDS_BYTES, &mut blobs[vi])
            .expect("launch");
        gpu.hip.event_record(&ev1, None).expect("rec");
        gpu.hip.event_synchronize(&ev1).expect("evsync");
        gpu.hip.event_elapsed_ms(&ev0, &ev1).expect("elapsed") as f64 * 1000.0
    };
    let n = active.len();
    for s in 0..WARMUP {
        for j in 0..n {
            time_one(gpu, active[(s + j) % n], &mut blobs);
        }
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); VARIANTS.len()];
    for s in 0..SAMPLES {
        for j in 0..n {
            let vi = active[(s + j) % n];
            let t = time_one(gpu, vi, &mut blobs);
            samples[vi].push(t);
        }
    }
    gpu.hip.event_destroy(ev0).expect("destroy");
    gpu.hip.event_destroy(ev1).expect("destroy");

    let mut median_us = vec![f64::NAN; VARIANTS.len()];
    println!(
        "  timing: {SAMPLES} individual launches/variant after {WARMUP} warmup, order cyclically \
         shifted per sample"
    );
    for &vi in active {
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

    for t in [x_t, n_t, y_t] {
        gpu.free_tensor(t).expect("free");
    }
    CaseOutcome {
        name: spec.name.clone(),
        rows,
        geo: spec.geo,
        ok,
        median_us,
    }
}

fn main() {
    let mut rows_override: Option<Vec<(usize, bool)>> = None;
    let mut variant_filter: Option<Vec<String>> = None;
    let mut filter: Option<String> = None;
    // `--precheck`: correctness pre-check on a gfx1100 / gfx1201 card (the same
    // kernels compile there): bitwise verdicts are meaningful, timings are not
    // gfx1151 numbers.
    let mut precheck = false;
    for a in std::env::args().skip(1) {
        if let Some(list) = a.strip_prefix("--rows=") {
            let v: Vec<(usize, bool)> = list
                .split(',')
                .map(|s| {
                    let s = s.trim();
                    let (num, bf) = match s.strip_suffix('b') {
                        Some(n) => (n, true),
                        None => (s, false),
                    };
                    (
                        num.parse()
                            .unwrap_or_else(|_| panic!("bad --rows entry {s:?}")),
                        bf,
                    )
                })
                .collect();
            assert!(
                !v.is_empty() && v.iter().all(|&(r, _)| r > 0),
                "--rows needs positive counts"
            );
            rows_override = Some(v);
        } else if let Some(list) = a.strip_prefix("--variants=") {
            variant_filter = Some(
                list.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            );
        } else if a == "--precheck" {
            precheck = true;
        } else if a.starts_with("--") {
            eprintln!("unknown flag {a} (supported: --rows=a,b,c, --variants=sx, --precheck)");
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
            "REFUSING: test_hc_read_up_twin_gfx1151 is a gfx1151-only oracle (found {}; \
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

    // The incumbent (index 0) always runs; twins are substring-matched on the symbol.
    let active: Vec<usize> = (0..VARIANTS.len())
        .filter(|&i| {
            i == 0
                || variant_filter.as_ref().map_or(true, |f| {
                    f.iter().any(|p| VARIANTS[i].func.contains(p.as_str()))
                })
        })
        .collect();
    if active.len() == 1 && variant_filter.is_some() {
        eprintln!("--variants matched no twin symbol");
        std::process::exit(2);
    }

    println!(
        "source sha256[..8]: incumbent {} (mq_fwht256.h + rope_yarn.h + tensor_ops.hip), twin {} \
         (hyper_read_up_fused_twin.gfx1151.hip)",
        sha_prefix(SRC_INCUMBENT),
        sha_prefix(SRC_TWIN)
    );
    for &i in &active {
        let v = &VARIANTS[i];
        gpu.ensure_kernel_public(v.module, v.src, v.func)
            .unwrap_or_else(|e| panic!("compile {}: {e:?}", v.func));
        println!("  loaded {:<9} {}", v.label, v.func);
    }

    // W: [4*hidden, K] BF16 bits, |w| in [1e-3, 0.1], random sign; shared by all cases.
    let w_t = {
        let mut rng = Rng(SEED ^ 0x5745_4947_4854);
        let bits: Vec<u16> = (0..WIDE * LOW_RANK)
            .map(|_| {
                let mag = 1e-3 + rng.unit() * (0.1 - 1e-3);
                let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
                bf16_bits((sign * mag) as f32)
            })
            .collect();
        gpu.upload_raw(&u16_bytes(&bits), &[bits.len() * 2])
            .expect("W upload")
    };

    let all_specs = specs(rows_override.as_deref());
    let mut outcomes = Vec::new();
    for sp in all_specs.iter() {
        if filter.as_deref().is_some_and(|f| !sp.name.contains(f)) {
            continue;
        }
        outcomes.push(run_case(&mut gpu, sp, &w_t, &active));
    }
    gpu.free_tensor(w_t).expect("free W");

    let mut all_ok = !outcomes.is_empty();
    println!("\n================ SUMMARY (median us; ratio = incumbent median / variant median) ================");
    print!(
        "{:<10} {:>6} {:>7} | {:>9}",
        "case", "rows", "verdict", "inc us"
    );
    for &i in active.iter().skip(1) {
        print!(" | {:>9} {:>6}", format!("{} us", VARIANTS[i].label), "x");
    }
    println!();
    for o in &outcomes {
        all_ok &= o.ok;
        print!(
            "{:<10} {:>6} {:>7} | {:>9.2}",
            o.name,
            o.rows,
            if o.ok { "PASS" } else { "FAIL" },
            o.median_us[0]
        );
        for &i in active.iter().skip(1) {
            print!(
                " | {:>9.2} {:>6.3}",
                o.median_us[i],
                o.median_us[0] / o.median_us[i]
            );
        }
        println!();
    }

    let geo: Vec<&CaseOutcome> = outcomes.iter().filter(|o| o.geo).collect();
    if geo.is_empty() {
        println!("\ngeomean ratio: no f32-normalized main cases ran");
    } else {
        println!(
            "\ngeometric-mean ratio (incumbent/variant median) over {} f32-normalized main cases:",
            geo.len()
        );
        for &i in active.iter().skip(1) {
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
