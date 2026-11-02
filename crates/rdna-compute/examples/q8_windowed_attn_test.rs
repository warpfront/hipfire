// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Correctness test for SLIDING-WINDOW masking in the batched Q8 flash attention
// (`attention_flash_q8_0_batched_masked_windowed`). Used by cohere2moe's
// `sliding_attention` layers so that, at context > sliding_window, a query at
// position p attends ONLY to keys in [p - window + 1, p] (the last `window`).
//
// No reference implementation needed — we assert three behavioral properties
// that together pin the masking semantics:
//
//   (1) DIFFERS:    windowed(W) != full-causal(W=0) when seq_len > W
//                   (the window actually clips something).
//   (2) INVARIANT:  overwriting the OUT-OF-WINDOW keys [0, seq_len-W) with huge
//                   garbage does NOT change the windowed output (they are truly
//                   excluded — the strongest check; a broken mask that leaks
//                   them would be dominated by the huge scores and diverge).
//   (3) EQUIVALENT: windowed(W >= seq_len) == full-causal (no clipping).
//
// Exit 0 = all pass; exit 1 = a property failed (the assertion message says
// which). Run: cargo run --release -p rdna-compute --example q8_windowed_attn_test

use rdna_compute::{DType, Gpu};

const NH: usize = 8;
const NKV: usize = 2;
const HD: usize = 128;
const S: usize = 384; // sequence length (3 tiles of 128)
const W: usize = 128; // sliding window
const BLK: usize = 34; // Q8_0 block: fp16 scale + 32 i8 codes
const TILE: usize = 128;

// Build a Q8_0 K/V cache row pattern; `huge` rows get large-magnitude codes so
// that if they leaked into a windowed softmax they would dominate it.
fn fill_kv(ctx: usize, huge_below: usize) -> Vec<u8> {
    let blocks_per_head = HD / 32;
    let bytes_per_pos = NKV * blocks_per_head * BLK;
    let mut kv = vec![0u8; ctx * bytes_per_pos];
    for pos in 0..ctx {
        let huge = pos < huge_below;
        for blk_i in 0..(NKV * blocks_per_head) {
            let off = pos * bytes_per_pos + blk_i * BLK;
            // fp16 scale: 1.0 (0x3C00) normally, ~100.0 (0x5640) for huge rows.
            let scale_bits: u16 = if huge { 0x5640 } else { 0x3C00 };
            kv[off] = (scale_bits & 0xFF) as u8;
            kv[off + 1] = (scale_bits >> 8) as u8;
            for j in 0..32 {
                // varied small codes normally; max (127) for huge rows.
                kv[off + 2 + j] = if huge {
                    127i8 as u8
                } else {
                    (((pos + blk_i + j) % 7) as i32 - 3) as i8 as u8
                };
            }
        }
    }
    kv
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let blocks_per_head = HD / 32;
    let bytes_per_pos = NKV * blocks_per_head * BLK;

    // Q: [1 × NH × HD], a varied pattern.
    let q_data: Vec<f32> = (0..NH * HD)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let q = gpu.upload_f32(&q_data, &[NH * HD]).expect("q");

    // positions: single query at the tail (pos = S-1).
    let pos_data: Vec<i32> = vec![(S - 1) as i32];
    let pos_bytes = unsafe { std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, 4) };
    let positions = gpu.upload_raw(pos_bytes, &[1]).expect("pos");

    let max_tiles = S.div_ceil(TILE);
    let partials = gpu
        .zeros(&[NH * max_tiles * (2 + HD)], DType::F32)
        .expect("partials");

    // Helper: run one windowed attention and return the host output vector.
    let mut run = |gpu: &mut Gpu, kv: &[u8], window: i32| -> Vec<f32> {
        let k = gpu.upload_raw(kv, &[kv.len()]).expect("k");
        let v = gpu.upload_raw(kv, &[kv.len()]).expect("v");
        let out = gpu.zeros(&[NH * HD], DType::F32).expect("out");
        gpu.attention_flash_q8_0_batched_masked_windowed(
            &q, &k, &v, &out, &positions, NH, NKV, HD, S, S, 1, &partials, None, 0, 0, window,
        )
        .expect("windowed attn launch");
        gpu.download_f32(&out).expect("download")
    };

    let kv_normal = fill_kv(S, 0);
    // Overwrite the OUT-OF-WINDOW region [0, S-W) with huge garbage.
    let kv_huge_outside = fill_kv(S, S - W);

    let out_full = run(&mut gpu, &kv_normal, 0); // full causal
    let out_win = run(&mut gpu, &kv_normal, W as i32); // windowed
    let out_win_huge = run(&mut gpu, &kv_huge_outside, W as i32); // windowed, huge outside
    let out_bigw = run(&mut gpu, &kv_normal, (S + 16) as i32); // window >= seq

    let d_clip = max_abs_diff(&out_full, &out_win);
    let d_invar = max_abs_diff(&out_win, &out_win_huge);
    let d_equiv = max_abs_diff(&out_full, &out_bigw);

    println!("seq={S} window={W} heads={NH} head_dim={HD}");
    println!("(1) clip   |full - win|            = {d_clip:.6}  (want > 0.001)");
    println!("(2) invar  |win  - win_huge_out|   = {d_invar:.6}  (want < 0.001)");
    println!("(3) equiv  |full - win(W>=seq)|    = {d_equiv:.6}  (want < 0.001)");

    let mut ok = true;
    if !(d_clip > 1e-3) {
        eprintln!(
            "FAIL (1): windowed output == full-causal; window had NO effect (masking not applied)"
        );
        ok = false;
    }
    if !(d_invar < 1e-3) {
        eprintln!(
            "FAIL (2): huge OUT-OF-WINDOW keys changed the output; they leaked past the mask"
        );
        ok = false;
    }
    if !(d_equiv < 1e-3) {
        eprintln!("FAIL (3): window>=seq differs from full causal; mask over-clips in-window keys");
        ok = false;
    }

    // ── Same three properties for the DECODE path (non-batched
    //    attention_flash_q8_0_windowed — a different kernel file). ──
    let mut run_nb = |gpu: &mut Gpu, kv: &[u8], window: i32| -> Vec<f32> {
        let k = gpu.upload_raw(kv, &[kv.len()]).expect("k");
        let v = gpu.upload_raw(kv, &[kv.len()]).expect("v");
        let out = gpu.zeros(&[NH * HD], DType::F32).expect("out");
        gpu.attention_flash_q8_0_windowed(
            &q,
            &k,
            &v,
            &out,
            &positions.buf,
            S,
            NH,
            NKV,
            HD,
            S,
            &partials,
            window,
        )
        .expect("nb windowed attn launch");
        gpu.download_f32(&out).expect("download")
    };
    let nb_full = run_nb(&mut gpu, &kv_normal, 0);
    let nb_win = run_nb(&mut gpu, &kv_normal, W as i32);
    let nb_win_huge = run_nb(&mut gpu, &kv_huge_outside, W as i32);
    let nb_bigw = run_nb(&mut gpu, &kv_normal, (S + 16) as i32);
    let nd_clip = max_abs_diff(&nb_full, &nb_win);
    let nd_invar = max_abs_diff(&nb_win, &nb_win_huge);
    let nd_equiv = max_abs_diff(&nb_full, &nb_bigw);
    println!("decode (non-batched): clip={nd_clip:.6} invar={nd_invar:.6} equiv={nd_equiv:.6}");
    if !(nd_clip > 1e-3) {
        eprintln!("FAIL decode(1): windowed == full; window had no effect");
        ok = false;
    }
    if !(nd_invar < 1e-3) {
        eprintln!("FAIL decode(2): huge out-of-window keys leaked past the mask");
        ok = false;
    }
    if !(nd_equiv < 1e-3) {
        eprintln!("FAIL decode(3): window>=seq over-clips in-window keys");
        ok = false;
    }

    // ── HD512 prefill/decode consistency (batched vs single-query) ──
    // Regression for the HD512 Q-preload/dot grouping in
    // `attention_flash_q8_0_tile_batched`: the batched kernel must preload Q
    // and group the Q·K dot as four 128-dim halves × four dims/thread —
    // exactly the single-query `attention_flash_q8_0_tile` association. The
    // contiguous 16-dim/thread grouping differs in FP association, so a
    // batched prefill row diverges from a single-query decode of the SAME
    // cache/query/position. One multirow batch covers BOS, a two-token
    // prefix, the 124-token reproducer length, a tile-boundary pair, and the
    // tail; each row is compared against `attention_flash_q8_0` (full
    // causal, window 0) on the same cache and query at the same position.
    {
        const NH5: usize = 4;
        const NKV5: usize = 2;
        const HD5: usize = 512;
        const BLK5: usize = 34; // Q8_0 block: fp16 scale + 32 i8 codes
                                // Derive — never guess: the batched launcher tiles by
                                // `Gpu::attn_tile_size`; the single-query path tiles by
                                // `attention::q8_flash_tile_size`.
        let tile_b = gpu.attn_tile_size();
        let s5: usize = (tile_b + 64).max(160);
        let tile_s = rdna_compute::attention::q8_flash_tile_size(&gpu.arch, NH5, NKV5, HD5, s5);
        // BOS, two-token prefix, 124-token reproducer length, tile-boundary
        // pair, tail.
        let pos5: [i32; 7] = [
            0,
            1,
            2,
            123,
            (tile_b - 1) as i32,
            tile_b as i32,
            (s5 - 1) as i32,
        ];
        let b5 = pos5.len();

        // Deterministic Q8_0 K/V cache: fp16 scale 1.0, varied small codes.
        let blocks5 = HD5 / 32;
        let bytes_per_pos5 = NKV5 * blocks5 * BLK5;
        let mut kv5 = vec![0u8; s5 * bytes_per_pos5];
        for pos in 0..s5 {
            for blk_i in 0..(NKV5 * blocks5) {
                let off = pos * bytes_per_pos5 + blk_i * BLK5;
                kv5[off] = 0x00;
                kv5[off + 1] = 0x3C; // fp16 scale 1.0
                for j in 0..32 {
                    kv5[off + 2 + j] =
                        (((pos * 31 + blk_i * 7 + j * 3) % 13) as i32 - 6) as i8 as u8;
                }
            }
        }

        // Deterministic, finite, nondegenerate Q, varied by row, head, dim.
        let q5_data: Vec<f32> = (0..b5 * NH5 * HD5)
            .map(|i| {
                let r = i / (NH5 * HD5);
                let h = (i / HD5) % NH5;
                let d = i % HD5;
                (((r * 7919 + h * 104729 + d * 1299709 + 12345) % 2001) as f32) / 1000.0 - 1.0
            })
            .collect();

        // Candidate: one multirow full-causal batch.
        let q5 = gpu.upload_f32(&q5_data, &[b5 * NH5 * HD5]).expect("q5");
        let pos5_bytes =
            unsafe { std::slice::from_raw_parts(pos5.as_ptr() as *const u8, pos5.len() * 4) };
        let positions5 = gpu.upload_raw(pos5_bytes, &[b5]).expect("pos5");
        let k5 = gpu.upload_raw(&kv5, &[kv5.len()]).expect("k5");
        let v5 = gpu.upload_raw(&kv5, &[kv5.len()]).expect("v5");
        let max_tiles_b = s5.div_ceil(tile_b);
        let partials5 = gpu
            .zeros(&[b5 * NH5 * max_tiles_b * (2 + HD5)], DType::F32)
            .expect("partials5");
        let out5 = gpu.zeros(&[b5 * NH5 * HD5], DType::F32).expect("out5");
        gpu.attention_flash_q8_0_batched_masked(
            &q5,
            &k5,
            &v5,
            &out5,
            &positions5,
            NH5,
            NKV5,
            HD5,
            s5,
            s5,
            b5,
            &partials5,
            None,
            0,
            0,
        )
        .expect("hd512 batched attn launch");
        let got5 = gpu.download_f32(&out5).expect("download5");

        // Reference: single-query decode per row on the SAME cache/query/position.
        let max_tiles_s = s5.div_ceil(tile_s);
        println!("hd512: batch={b5} heads={NH5} kv={NKV5} dim={HD5} seq={s5} tile_b={tile_b} tile_s={tile_s}");
        for (r, &p) in pos5.iter().enumerate() {
            let row_q = &q5_data[r * NH5 * HD5..(r + 1) * NH5 * HD5];
            let qr = gpu.upload_f32(row_q, &[NH5 * HD5]).expect("q5 row");
            let pd = [p];
            let pdb = unsafe { std::slice::from_raw_parts(pd.as_ptr() as *const u8, 4) };
            let post = gpu.upload_raw(pdb, &[1]).expect("pos row");
            let outr = gpu.zeros(&[NH5 * HD5], DType::F32).expect("out row");
            let partr = gpu
                .zeros(&[NH5 * max_tiles_s * (2 + HD5)], DType::F32)
                .expect("partials row");
            gpu.attention_flash_q8_0(
                &qr,
                &k5,
                &v5,
                &outr,
                &post.buf,
                (p + 1) as usize,
                NH5,
                NKV5,
                HD5,
                s5,
                &partr,
            )
            .expect("hd512 single attn launch");
            let want = gpu.download_f32(&outr).expect("download row");
            let got_row = &got5[r * NH5 * HD5..(r + 1) * NH5 * HD5];
            for (i, (&g, &w)) in got_row.iter().zip(want.iter()).enumerate() {
                if !g.is_finite() || !w.is_finite() {
                    eprintln!(
                        "FAIL hd512: row {r} pos {p} dim {i} non-finite (batched={g} single={w})"
                    );
                    ok = false;
                }
            }
            let d = max_abs_diff(got_row, &want);
            // Rows inside the first tile of BOTH paths are tiling-independent
            // (single tile → identical reduction math), so they always demand
            // exact numerical equality (d == 0; signed zero accepts). Later
            // rows demand it too when both resolvers agree (the lab default);
            // where the two tile sizes disagree the tiling itself rounds
            // differently, so those rows use a tight tolerance instead.
            let single_tile_row = (p + 1) as usize <= tile_b.min(tile_s);
            if single_tile_row || tile_b == tile_s {
                println!("hd512: row {r} pos {p} |batched - single| = {d:.6e} (want exactly 0)");
                if d != 0.0 {
                    eprintln!(
                        "FAIL hd512: row {r} pos {p} prefill/decode mismatch: batched vs single-query differ by {d:.6e}"
                    );
                    ok = false;
                }
            } else {
                println!("hd512: row {r} pos {p} |batched - single| = {d:.6e} (tiles differ; want < 1e-5)");
                if !(d <= 1e-5) {
                    eprintln!(
                        "FAIL hd512: row {r} pos {p} differs from single-query decode by {d:.6e}"
                    );
                    ok = false;
                }
            }
        }
        println!("hd512: prefill/decode consistency checked ({b5} rows)");
    }

    if ok {
        println!("PASS: sliding-window masking correct (prefill + decode)");
    } else {
        std::process::exit(1);
    }
}
