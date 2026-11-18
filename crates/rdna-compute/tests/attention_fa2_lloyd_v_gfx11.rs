// SPDX-License-Identifier: Apache-2.0

//! gfx11 FA2 lloyd-V prefill combos against the scalar lloyd-V path.
//!
//! `attention_q8_0_fa2_gqa_fwht4k_lloydv4_gfx11` (fwht4 K + lloyd4 V) and
//! `attention_q8_0_fa2_gqa_fwht3k_lloydv3_gfx11` (fwht3 K + lloyd3 V) must
//! produce the same O as `attention_flash_fwht{4,3}_batched_masked` with
//! `v_mode_bits` 4 / 3 on the same caches, Q, positions and sign tables.
//! FA2 runs QK and PV in f16, the scalar path in f32, so the comparison is a
//! relative max-abs tolerance, not bit equality. Shapes cover batch 64 and 65
//! (a partial 16-row query tile) and causal spans that are not a multiple of
//! the 32-key tile.
//!
//! Negative control: the FA2 output with a signed FWHT-256 applied again
//! (rotated space, the shape of error a dropped
//! `attention_fa2_o_postinvert_lloyd_gfx11` launch leaves in `out`) must miss
//! the scalar output by more than 10x the tolerance.
//!
//! `#[ignore]`d: needs a gfx11 GPU (gfx1100/1101/1102/1150/1151) and hipcc.
//!
//!   cargo test -p rdna-compute --release --test attention_fa2_lloyd_v_gfx11 -- --ignored

use rdna_compute::{gen_fwht_signs, DType, Gpu, GpuTensor};

const N_HEADS: usize = 24;
const N_KV: usize = 4;
const HD: usize = 256;
const MAX_SEQ: usize = 2048;
/// Measured FA2-vs-scalar relative max-abs error is f16-rounding sized
/// (see the printed values); a missing post-inverse or a wrong V record
/// offset is O(1) relative.
const TOL: f32 = 2e-2;

/// `(batch, first position)`: the causal span of the last row is
/// `first + batch` keys (1064, 1065, 102, 1065), none a multiple of 32.
const SHAPES: &[(usize, usize)] = &[(64, 1000), (65, 1000), (65, 37), (64, 1001)];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFFFF) as f32 / 32768.0 - 1.0
    }
}

/// `[pos][kv head][f32 cnorm][payload]` records (fwht K and lloyd V share it:
/// 132 B/head for 4-bit, 100 B/head for 3-bit) with random codes and a
/// per-record norm in [2, 10).
fn records(rng: &mut Rng, head_bytes: usize) -> Vec<u8> {
    let mut b = Vec::with_capacity(MAX_SEQ * N_KV * head_bytes);
    for _ in 0..MAX_SEQ * N_KV {
        let cnorm = 2.0 + 8.0 * (rng.next() % 4096) as f32 / 4096.0;
        b.extend_from_slice(&cnorm.to_le_bytes());
        for _ in 4..head_bytes {
            b.push(rng.next() as u8);
        }
    }
    b
}

fn rel_max_diff(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-30);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / scale
}

/// Signed FWHT-256 (`x * s1 -> H -> * s2 / 16`): an orthonormal rotation
/// of the same family as the V rotation, used only by the negative control.
fn fwht_forward_256(row: &mut [f32], s1: &[f32], s2: &[f32]) {
    for (x, s) in row.iter_mut().zip(s1) {
        *x *= s;
    }
    let mut h = 1;
    while h < HD {
        for i in (0..HD).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (row[j], row[j + h]);
                row[j] = a + b;
                row[j + h] = a - b;
            }
        }
        h *= 2;
    }
    for (x, s) in row.iter_mut().zip(s2) {
        *x *= s / 16.0;
    }
}

fn upload_bytes(gpu: &Gpu, b: &[u8]) -> GpuTensor {
    gpu.upload_raw(b, &[b.len()]).expect("upload")
}

#[test]
#[ignore = "needs a gfx11 GPU and hipcc"]
fn fa2_lloyd_v_matches_scalar_lloyd_v() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(
        matches!(
            gpu.arch.as_str(),
            "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151"
        ),
        "gfx11 only, got {}",
        gpu.arch
    );
    let s1h = gen_fwht_signs(42, HD);
    let s2h = gen_fwht_signs(1042, HD);
    let s1 = gpu.upload_f32(&s1h, &[HD]).expect("s1");
    let s2 = gpu.upload_f32(&s2h, &[HD]).expect("s2");
    let mut rng = Rng(0x6c6c_6f79_6476);
    let tile = gpu.attn_tile_size();
    for (bits, head_bytes) in [(4i32, 132usize), (3, 100)] {
        let kb = records(&mut rng, head_bytes);
        let vb = records(&mut rng, head_bytes);
        let k = upload_bytes(&gpu, &kb);
        let v = upload_bytes(&gpu, &vb);
        for &(batch, first) in SHAPES {
            let ctx = first + batch;
            assert!(ctx % 32 != 0 && ctx <= MAX_SEQ);
            let pos: Vec<u8> = (0..batch)
                .flat_map(|b| ((first + b) as i32).to_le_bytes())
                .collect();
            let positions = upload_bytes(&gpu, &pos);
            let qh: Vec<f32> = (0..batch * N_HEADS * HD)
                .map(|_| 3.0 * rng.unit())
                .collect();
            let q_fa2 = gpu.upload_f32(&qh, &[qh.len()]).expect("q");
            let q_ref = gpu.upload_f32(&qh, &[qh.len()]).expect("q");
            let o_fa2 = gpu.zeros(&[qh.len()], DType::F32).expect("o");
            let o_ref = gpu.zeros(&[qh.len()], DType::F32).expect("o");
            let partials = gpu
                .zeros(
                    &[batch * N_HEADS * ctx.div_ceil(tile) * (2 + HD)],
                    DType::F32,
                )
                .expect("partials");
            if bits == 4 {
                gpu.attention_q8_0_fa2_gqa_fwht4k_lloydv4_gfx11(
                    &q_fa2, &k, &v, &o_fa2, &positions, &s1, &s2, N_HEADS, N_KV, HD, ctx, batch,
                )
                .expect("fa2 (4,4)");
                gpu.attention_flash_fwht4_batched_masked(
                    &q_ref, &k, &v, &o_ref, &positions, &s1, &s2, N_HEADS, N_KV, HD, MAX_SEQ, ctx,
                    batch, &partials, None, 0, 0, 4,
                )
                .expect("scalar fwht4/lloyd4");
            } else {
                gpu.attention_q8_0_fa2_gqa_fwht3k_lloydv3_gfx11(
                    &q_fa2, &k, &v, &o_fa2, &positions, &s1, &s2, N_HEADS, N_KV, HD, ctx, batch,
                )
                .expect("fa2 (3,3)");
                gpu.attention_flash_fwht3_batched_masked(
                    &q_ref, &k, &v, &o_ref, &positions, &s1, &s2, N_HEADS, N_KV, HD, MAX_SEQ, ctx,
                    batch, &partials, None, 0, 0, 3,
                )
                .expect("scalar fwht3/lloyd3");
            }
            let fa2 = gpu.download_f32(&o_fa2).expect("dl");
            let want = gpu.download_f32(&o_ref).expect("dl");
            assert!(fa2.iter().all(|x| x.is_finite()), "non-finite FA2 output");
            let d = rel_max_diff(&fa2, &want);
            let mut rotated = fa2.clone();
            for row in rotated.chunks_mut(HD) {
                fwht_forward_256(row, &s1h, &s2h);
            }
            let d_neg = rel_max_diff(&rotated, &want);
            println!(
                "lloyd{bits} batch={batch} first={first} ctx={ctx}: rel|fa2-scalar|={d:.2e} \
                 neg(no post-inverse)={d_neg:.2e}"
            );
            assert!(d < TOL, "lloyd{bits} batch={batch} ctx={ctx}: {d} >= {TOL}");
            assert!(
                d_neg > 10.0 * TOL,
                "lloyd{bits} batch={batch} ctx={ctx}: negative control {d_neg}"
            );
        }
    }
}
