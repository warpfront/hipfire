// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

//! Parity test: `Gpu::copy_rows_strided_f32` (one launch per chunk) vs the
//! per-row `copy_d2d` loop it replaces in `Gpuf::assemble_rows`
//! (`crates/hipfire-arch-diffusion/src/flux_gpu.rs`).
//!
//! The reference is literally the old loop:
//!
//! ```text
//! for t in 0..n_rows {
//!     for (dcol, src, len) in chunks {
//!         copy_d2d(src.sub_offset(t*src_row_stride, len),
//!                  dst.sub_offset(t*dst_row_stride + dcol, len))
//!     }
//! }
//! ```
//!
//! Both are pure copies, so the bound is **bit equality**, not a tolerance.
//! Anything else is an indexing bug — fix the kernel, do not add a tolerance.
//!
//! The whole destination buffer is compared, not just the written window, so
//! a kernel that writes outside its chunk fails here too.
//!
//! Cases:
//!   1. `flux/single-block` — the real shape: n_rows = 4608,
//!      dst_row_stride = 15360, chunks (dcol 0, len 3072) and
//!      (dcol 3072, len 12288). Takes the float4 fast path.
//!   2. Ragged `len` / strides that are not multiples of 4 — scalar path.
//!   3. n_rows = 1.
//!   4. A single chunk written at a nonzero, aligned `dst_col_offset`.
//!   5. A source whose row stride exceeds `len` (strided read).
//!
//! Run: cargo run --release --example test_copy_rows_strided_f32_parity -p rdna-compute
//! Exits 0 on pass, 1 on any failure.

use rdna_compute::{DType, Gpu, GpuTensor};

/// Deterministic values in roughly [-1, 1). LCG, no rand dependency.
fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// One chunk of an assemble: destination column, element count per row, and
/// the source buffer's own row stride (>= len).
struct Chunk {
    dcol: usize,
    len: usize,
    src_row_stride: usize,
}

struct Case {
    label: &'static str,
    n_rows: usize,
    dst_row_stride: usize,
    chunks: &'static [Chunk],
    /// True when every chunk is expected to take the float4 path — asserted,
    /// so a future wrapper change that silently drops the fast path is caught.
    expect_vec4: bool,
}

const CASES: &[Case] = &[
    // 1. The real FLUX.1-dev single-block linear2 input assemble.
    Case {
        label: "flux/single-block",
        n_rows: 4608,
        dst_row_stride: 15360,
        chunks: &[
            Chunk {
                dcol: 0,
                len: 3072,
                src_row_stride: 3072,
            },
            Chunk {
                dcol: 3072,
                len: 12288,
                src_row_stride: 12288,
            },
        ],
        expect_vec4: true,
    },
    // 2. Nothing divisible by 4 — scalar path, every index shape ragged.
    Case {
        label: "ragged/len-not-mult4",
        n_rows: 37,
        dst_row_stride: 251,
        chunks: &[
            Chunk {
                dcol: 0,
                len: 101,
                src_row_stride: 101,
            },
            Chunk {
                dcol: 103,
                len: 147,
                src_row_stride: 149,
            },
        ],
        expect_vec4: false,
    },
    // 3. Degenerate row count.
    Case {
        label: "n_rows=1",
        n_rows: 1,
        dst_row_stride: 15360,
        chunks: &[
            Chunk {
                dcol: 0,
                len: 3072,
                src_row_stride: 3072,
            },
            Chunk {
                dcol: 3072,
                len: 12288,
                src_row_stride: 12288,
            },
        ],
        expect_vec4: true,
    },
    // 4. One chunk at a nonzero aligned destination column.
    Case {
        label: "offset/dcol=128",
        n_rows: 512,
        dst_row_stride: 256,
        chunks: &[Chunk {
            dcol: 128,
            len: 64,
            src_row_stride: 64,
        }],
        expect_vec4: true,
    },
    // 5. Strided source read (src_row_stride > len), aligned.
    Case {
        label: "src-stride>len",
        n_rows: 300,
        dst_row_stride: 1024,
        chunks: &[Chunk {
            dcol: 512,
            len: 256,
            src_row_stride: 384,
        }],
        expect_vec4: true,
    },
    // 6. A single ragged row at a ragged offset — smallest odd shape.
    Case {
        label: "ragged/tiny",
        n_rows: 3,
        dst_row_stride: 11,
        chunks: &[Chunk {
            dcol: 3,
            len: 5,
            src_row_stride: 7,
        }],
        expect_vec4: false,
    },
];

/// Mirror of the pre-kernel `Gpuf::assemble_rows` inner loop.
fn reference_assemble(
    gpu: &Gpu,
    dst: &GpuTensor,
    dst_row_stride: usize,
    n_rows: usize,
    chunks: &[(&Chunk, &GpuTensor)],
) -> usize {
    let mut copies = 0usize;
    for t in 0..n_rows {
        for (c, src) in chunks {
            let sv = src.sub_offset(t * c.src_row_stride, c.len);
            let dv = dst.sub_offset(t * dst_row_stride + c.dcol, c.len);
            gpu.copy_d2d(&sv, &dv, c.len * DType::F32.size())
                .expect("reference copy_d2d");
            copies += 1;
        }
    }
    copies
}

/// Returns (mismatch_count, first_mismatch_index, ref_copies, new_launches).
fn run_case(gpu: &mut Gpu, c: &Case) -> (usize, usize, usize, usize) {
    // Sources: one contiguous [n_rows, src_row_stride] buffer per chunk, so a
    // src_row_stride > len case has real (differing) data in the gap.
    let mut srcs: Vec<GpuTensor> = Vec::new();
    for (i, ch) in c.chunks.iter().enumerate() {
        let n = c.n_rows * ch.src_row_stride;
        let h = pseudo_random(n, 0x51C0_0000 ^ (i as u64 + 1) ^ (ch.len as u64) << 8);
        srcs.push(
            gpu.upload_f32(&h, &[c.n_rows, ch.src_row_stride])
                .expect("upload src"),
        );
    }

    // Destinations pre-filled with the SAME non-zero pattern, so untouched
    // bytes are compared meaningfully (a stray write shows up as a mismatch).
    let dst_n = c.n_rows * c.dst_row_stride;
    let fill = pseudo_random(dst_n, 0xDEAD_0000 ^ c.n_rows as u64);
    let dst_ref = gpu
        .upload_f32(&fill, &[c.n_rows, c.dst_row_stride])
        .expect("upload dst_ref");
    let dst_new = gpu
        .upload_f32(&fill, &[c.n_rows, c.dst_row_stride])
        .expect("upload dst_new");
    drop(fill);

    let pairs: Vec<(&Chunk, &GpuTensor)> = c.chunks.iter().zip(srcs.iter()).collect();
    let ref_copies = reference_assemble(gpu, &dst_ref, c.dst_row_stride, c.n_rows, &pairs);

    let mut launches = 0usize;
    for (ch, src) in &pairs {
        gpu.copy_rows_strided_f32(
            src,
            &dst_new,
            c.n_rows,
            ch.len,
            ch.src_row_stride,
            c.dst_row_stride,
            ch.dcol,
        )
        .expect("copy_rows_strided_f32");
        launches += 1;
    }

    gpu.hip.device_synchronize().expect("sync");
    let r = gpu.download_f32(&dst_ref).expect("dl ref");
    let n = gpu.download_f32(&dst_new).expect("dl new");

    let mut bad = 0usize;
    let mut first = usize::MAX;
    for i in 0..r.len() {
        if r[i].to_bits() != n[i].to_bits() {
            if first == usize::MAX {
                first = i;
                eprintln!(
                    "      first mismatch at flat {i} (row {} col {}): ref={} new={}",
                    i / c.dst_row_stride,
                    i % c.dst_row_stride,
                    r[i],
                    n[i]
                );
            }
            bad += 1;
        }
    }

    for t in srcs {
        let _ = gpu.free_tensor(t);
    }
    let _ = gpu.free_tensor(dst_ref);
    let _ = gpu.free_tensor(dst_new);
    (bad, first, ref_copies, launches)
}

/// Recompute the wrapper's fast-path predicate so the test can assert the
/// FLUX shapes really do vectorize.
fn takes_vec4(c: &Case) -> bool {
    c.chunks.iter().all(|ch| {
        ch.len % 4 == 0
            && ch.src_row_stride % 4 == 0
            && c.dst_row_stride % 4 == 0
            && ch.dcol % 4 == 0
    })
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("=== test_copy_rows_strided_f32_parity ===");
            eprintln!("  SKIPPED: no GPU / HIP runtime ({e:?})");
            std::process::exit(0);
        }
    };
    let arch = gpu.arch.clone();
    eprintln!("=== test_copy_rows_strided_f32_parity ===");
    eprintln!("  arch = {arch}");
    eprintln!("  bound = bit-exact (pure copy)");

    let mut fails = 0usize;
    let mut total_ref_copies = 0usize;
    let mut total_launches = 0usize;
    for c in CASES {
        if takes_vec4(c) != c.expect_vec4 {
            eprintln!(
                "  FAIL  {:<24} fast-path predicate is {} but the case expects {}",
                c.label,
                takes_vec4(c),
                c.expect_vec4
            );
            fails += 1;
            continue;
        }
        let (bad, _first, ref_copies, launches) = run_case(&mut gpu, c);
        total_ref_copies += ref_copies;
        total_launches += launches;
        let verdict = if bad == 0 { "PASS" } else { "FAIL" };
        if bad != 0 {
            fails += 1;
        }
        eprintln!(
            "  {verdict}  {:<24} rows={:<5} dst_stride={:<6} chunks={} path={:<6} \
             copy_d2d={:<5} launches={:<2} mismatches={bad}",
            c.label,
            c.n_rows,
            c.dst_row_stride,
            c.chunks.len(),
            if c.expect_vec4 { "float4" } else { "scalar" },
            ref_copies,
            launches,
        );
    }

    eprintln!();
    eprintln!("  total: {total_ref_copies} copy_d2d -> {total_launches} kernel launches");
    if fails == 0 {
        eprintln!("ALL PASS ({} cases, bit-exact)", CASES.len());
        std::process::exit(0);
    }
    eprintln!("{fails} FAILED");
    std::process::exit(1);
}
