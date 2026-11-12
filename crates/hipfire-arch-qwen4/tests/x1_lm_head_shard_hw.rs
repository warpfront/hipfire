// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! X1 R1: the vocab-sharded LM head projects every row exactly as the whole head does.
//!
//! The head is the served one: MQ6G256V2, 248,320 x 2,560, 200-byte groups
//! (`[s0 z0 s1 z1: f16][192 B six-bit payload]`, 2,000 B a row). `program::execute_lm_head`
//! runs over (a) the whole head into `[rows][V]` and (b) a `WeightRef` that is the byte
//! sub-tensor of rows `[0, V/2)` and `[V/2, V)` (`m = 124,160`, same `row_stride`) into packed
//! `[rows][V/2]` buffers, with the same hidden rows and the same rotation-scratch semantics
//! (`rows * k` F32, rotated by the projection itself). Every row's bytes must match, for
//! rows in {1, 2, 3, 4, 5, 8, 9, 16, 24, 40, 48, 64}; the 64 floats after each packed output are
//! guard words that must stay untouched.
//!
//! Two weight flavours, because above 8 rows the route depends on how the loader marked the head:
//! * `plain`: affine headers, no trunk-A8 verdict: rows 9.. take the exact-INT8 MQ6 GEMM
//!   (`mq6_i8_route`: gfx1201 / gfx1151).
//! * `sym`: qt47 symmetric headers (`z == -32 * s` exactly), `Gpu::qwen4_mq6_sym_check` passes and
//!   the loader's verdict `set_qwen4_trunk_a8_verified(true)` is recorded: rows 9..=511 take
//!   `gemm_mq6sym_i8` (`mq6_a8_route`, exact gfx1201 only).
//!
//! `qwen4_scope = true` as the Qwen4 forward sets it. The dispatcher keeps its route predicates
//! private, so the test asserts the public ones it is built from (`qwen4_trunk_a8_applies`,
//! `gemm_mq6g256v2_i8_applies`, `few_row_limit`) for the full and half shapes and prints the
//! kernel entry each shape selects (`gemm_mq6sym_i8_kernel_name` /
//! `gemm_mq6g256v2_i8_kernel_name`); the full and half entries may differ (tile choice reads `m`):
//! the bytes may not.
//!
//! `#[ignore]`d: needs an exact gfx1201 (both flavours) or gfx1151 (`plain` only) GPU with about
//! 1.5 GB free VRAM.
//!
//!   cargo test -p hipfire-arch-qwen4 --release --test x1_lm_head_shard_hw -- --ignored --nocapture

use hipfire_arch_qwen4::program::{execute_lm_head, validate_lm_head};
use hipfire_dispatch::families::gemv::WeightRef;
use rdna_compute::{DType, Gpu, GpuTensor};

const VOCAB: usize = 248_320;
const HALF: usize = VOCAB / 2;
const K: usize = 2560;
const GROUP_ELEMS: usize = 256;
const GROUP_BYTES: usize = 200;
const ROW_STRIDE: usize = K / GROUP_ELEMS * GROUP_BYTES;
const ROWS: [usize; 12] = [1, 2, 3, 4, 5, 8, 9, 16, 24, 40, 48, 64];
const MAX_ROWS: usize = 64;
const GUARD: usize = 64;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Normal f16 bit pattern with biased exponent in `lo..=hi`.
    fn f16_normal(&mut self, lo: u64, hi: u64, signed: bool) -> u16 {
        let r = self.next_u64();
        let exp = lo + (r % (hi - lo + 1));
        let mant = (r >> 16) & 0x3FF;
        let sign = if signed { (r >> 40) & 1 } else { 0 };
        ((sign << 15) | (exp << 10) | mant) as u16
    }
}

/// `VOCAB` rows of MQ6G256V2 groups; every header a finite normal f16 (scales 2^-7..2^-2).
/// `sym`: `z = -32 * s` exactly (exponent + 5, sign set), the qt47 symmetric grid.
fn synthetic_mq6_head(sym: bool) -> Vec<u8> {
    let groups = VOCAB * (K / GROUP_ELEMS);
    let mut out = vec![0u8; groups * GROUP_BYTES];
    let mut rng = Rng(if sym {
        0x5eed_5eed_0001
    } else {
        0x5eed_5eed_0002
    });
    for g in 0..groups {
        let off = g * GROUP_BYTES;
        for half in 0..2 {
            let s = rng.f16_normal(8, 13, false);
            let z = if sym {
                ((s & 0x7fff) + (5 << 10)) | 0x8000
            } else {
                rng.f16_normal(8, 14, true)
            };
            out[off + 4 * half..off + 4 * half + 2].copy_from_slice(&s.to_le_bytes());
            out[off + 4 * half + 2..off + 4 * half + 4].copy_from_slice(&z.to_le_bytes());
        }
        for w in 0..24 {
            let v = rng.next_u64();
            out[off + 8 + 8 * w..off + 16 + 8 * w].copy_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn weight<'a>(buf: &'a GpuTensor, m: usize) -> WeightRef<'a> {
    WeightRef {
        buf,
        dtype: DType::MQ6G256V2,
        m,
        k: K,
        row_stride: ROW_STRIDE,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

fn bits(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.download_f32(t)
        .expect("download")
        .iter()
        .map(|f| f.to_bits())
        .collect()
}

fn describe(gpu: &Gpu, m: usize, rows: usize, y: &GpuTensor) -> String {
    format!(
        "m {m} rows {rows}: route_rows {} few_row_limit {} trunk_a8 {} i8 {} | sym entry {} | i8 entry {:?}",
        gpu.route_rows(rows),
        gpu.few_row_limit(),
        gpu.qwen4_trunk_a8_applies(m, K, rows),
        gpu.gemm_mq6g256v2_i8_applies(m, K, rows),
        gpu.gemm_mq6sym_i8_kernel_name(m, K, rows, false),
        gpu.gemm_mq6g256v2_i8_kernel_name(m, K, rows, y),
    )
}

fn run(sym: bool) {
    let mut gpu = Gpu::init().expect("GPU");
    let name = if sym { "sym" } else { "plain" };
    match (gpu.arch.as_str(), sym) {
        ("gfx1201", _) | ("gfx1151", false) => {}
        (arch, _) => {
            eprintln!("skip ({name}): {arch} has no such MQ6 head route");
            return;
        }
    }
    gpu.qwen4_scope = true;

    let bytes = synthetic_mq6_head(sym);
    let head = gpu.upload_raw(&bytes, &[bytes.len()]).expect("head upload");
    drop(bytes);
    let half_bytes = HALF * ROW_STRIDE;
    let head0 = head.sub_offset(0, half_bytes);
    let head1 = head.sub_offset(half_bytes, half_bytes);
    if sym {
        for (label, t, m) in [
            ("full", &head, VOCAB),
            ("rows [0,V/2)", &head0, HALF),
            ("rows [V/2,V)", &head1, HALF),
        ] {
            assert!(
                gpu.qwen4_mq6_sym_check(t, m, K).expect("sym check"),
                "the synthetic {label} head is not symmetric"
            );
        }
        gpu.set_qwen4_trunk_a8_verified(true);
    }
    let full = weight(&head, VOCAB);
    let half0 = weight(&head0, HALF);
    let half1 = weight(&head1, HALF);

    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let hidden_host: Vec<f32> = (0..MAX_ROWS * K)
        .map(|_| (rng.next_u64() & 0xFF_FFFF) as f32 / (1u32 << 24) as f32 - 0.5)
        .collect();
    let hidden = gpu
        .upload_f32(&hidden_host, &[MAX_ROWS * K])
        .expect("hidden");
    let rotation = gpu.zeros(&[MAX_ROWS * K], DType::F32).expect("rotation");
    let nan = |n: usize| vec![f32::from_bits(0x7fc0_dead); n];
    let full_out = gpu
        .upload_f32(&nan(MAX_ROWS * VOCAB), &[MAX_ROWS * VOCAB])
        .expect("full logits");
    let out0 = gpu
        .upload_f32(&nan(MAX_ROWS * HALF + GUARD), &[MAX_ROWS * HALF + GUARD])
        .expect("half 0 logits");
    let out1 = gpu
        .upload_f32(&nan(MAX_ROWS * HALF + GUARD), &[MAX_ROWS * HALF + GUARD])
        .expect("half 1 logits");

    for &rows in &ROWS {
        let x = hidden.sub_offset(0, rows * K);
        let rot = rotation.sub_offset(0, rows * K);
        let y_full = full_out.sub_offset(0, rows * VOCAB);
        let y0 = out0.sub_offset(0, rows * HALF);
        let y1 = out1.sub_offset(0, rows * HALF);

        eprintln!("[{name}] full: {}", describe(&gpu, VOCAB, rows, &y_full));
        eprintln!("[{name}] half: {}", describe(&gpu, HALF, rows, &y0));
        if rows > 8 {
            if sym {
                assert!(
                    gpu.qwen4_trunk_a8_applies(VOCAB, K, rows)
                        && gpu.qwen4_trunk_a8_applies(HALF, K, rows),
                    "{name} rows {rows}: the A8 trunk route is not taken (full and half must both take it)"
                );
            } else {
                // The MQ6 INT8 default scope is route_rows >= 512 on beta; what
                // matters here is that full and half take the same route.
                assert!(
                    !gpu.qwen4_trunk_a8_applies(VOCAB, K, rows)
                        && !gpu.qwen4_trunk_a8_applies(HALF, K, rows)
                        && gpu.gemm_mq6g256v2_i8_applies(VOCAB, K, rows)
                            == gpu.gemm_mq6g256v2_i8_applies(HALF, K, rows),
                    "{name} rows {rows}: full and half take different MQ6 routes"
                );
            }
        }

        validate_lm_head(&full, &x, &y_full, rows, rows).expect("validate full");
        validate_lm_head(&half0, &x, &y0, rows, rows).expect("validate half 0");
        execute_lm_head(&mut gpu, &full, &x, &y_full, rows, rows, Some(&rot)).expect("full head");
        execute_lm_head(&mut gpu, &half0, &x, &y0, rows, rows, Some(&rot)).expect("half 0 head");
        execute_lm_head(&mut gpu, &half1, &x, &y1, rows, rows, Some(&rot)).expect("half 1 head");
        gpu.hip.device_synchronize().expect("sync");

        let f = gpu.download_f32(&y_full).expect("full");
        let h0 = gpu.download_f32(&y0).expect("half 0");
        let h1 = gpu.download_f32(&y1).expect("half 1");
        assert!(
            f.iter().all(|v| v.is_finite()) && f.iter().any(|v| *v != 0.0),
            "{name} rows {rows}: the reference logits are degenerate"
        );
        for r in 0..rows {
            let want0 = &f[r * VOCAB..r * VOCAB + HALF];
            let want1 = &f[r * VOCAB + HALF..(r + 1) * VOCAB];
            for (half, want, got) in [
                (0, want0, &h0[r * HALF..(r + 1) * HALF]),
                (1, want1, &h1[r * HALF..(r + 1) * HALF]),
            ] {
                let bad = want
                    .iter()
                    .zip(got)
                    .position(|(a, b)| a.to_bits() != b.to_bits());
                assert!(
                    bad.is_none(),
                    "{name} rows {rows}: row {r} half {half} differs at logit {:?}: whole {:#x} sharded {:#x}",
                    bad,
                    want[bad.unwrap_or(0)].to_bits(),
                    got[bad.unwrap_or(0)].to_bits()
                );
            }
        }
        // Guard words after each packed output stay untouched.
        for (label, out) in [("half 0", &out0), ("half 1", &out1)] {
            let tail = bits(&gpu, &out.sub_offset(rows * HALF, GUARD));
            assert!(
                tail.iter().all(|&b| b == 0x7fc0_dead),
                "{name} rows {rows}: {label} wrote past its packed [rows][V/2] output"
            );
        }
        eprintln!(
            "[{name}] rows {rows}: {} logits per row byte-identical (whole vs two halves)",
            VOCAB
        );
    }
}

#[test]
#[ignore = "needs an exact gfx1201 (or gfx1151 for plain) GPU: run through the GPU fabric"]
fn lm_head_halves_equal_whole_head_plain_mq6() {
    run(false);
}

#[test]
#[ignore = "needs an exact gfx1201 GPU: run through the GPU fabric"]
fn lm_head_halves_equal_whole_head_sym_qt47_a8() {
    run(true);
}
