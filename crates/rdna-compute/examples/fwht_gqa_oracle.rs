// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! fwht4 GQA tile vs scalar tile differential on synthetic inputs. Same
//! q/K/V/signs/positions go into `attention_flash_fwht4` (scalar) and
//! `attention_flash_fwht4_tile_gqa_gfx11` (GQA, Q8_0 V); the
//! `[n_heads, max_tiles, 2+256]` partials must be bit-identical, otherwise
//! the oracle panics naming the first differing (head, tile, dim) and the
//! max abs diff. Inputs are randomized from a fixed-seed LCG: per-record K
//! `cnorm` in [0.5, 4), Q8_0 f16 scales in [2^-3, 2) and +-1 sign tables.
//! Needs a gfx1100 device. Run with `--features lab`.
//!
//! Run: `cargo run -p rdna-compute --features lab --example fwht_gqa_oracle`

use rdna_compute::{DType, Gpu};

const N_HEADS: usize = 24;
const N_KV: usize = 4;
const HD: usize = 256;

fn main() {
    let mut found = None;
    for ordinal in 0..8 {
        match Gpu::init_with_device(ordinal) {
            Ok(g) => {
                if g.arch == "gfx1100" {
                    found = Some(g);
                    break;
                }
            }
            Err(_) => break,
        }
    }
    let mut gpu = found.expect("no gfx1100 among HIP ordinals");
    eprintln!("oracle arch: {}", gpu.arch);

    for t in [1usize, 2, 16, 17, 32, 128, 256, 66048] {
        eprintln!("=== T={t} ===");
        // The launcher needs max_seq >= T and partials sized for max_tiles.
        let max_seq = ((t + 127) / 128 * 128).max(512);

        let mut s = 0x12345678u64;
        let mut rnd = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u32
        };
        let unit = |r: &mut dyn FnMut() -> u32| r() as f32 / (1u64 << 31) as f32; // [0, 1)

        // q: random f32 in [-1, 1).
        let mut qh = vec![0f32; N_HEADS * HD];
        for v in qh.iter_mut() {
            *v = (unit(&mut rnd) - 0.5) * 2.0;
        }
        // K: fwht4 records [f32 cnorm in [0.5, 4)][128 random bytes] per (pos, kv).
        let mut kh = vec![0u8; t * N_KV * 132];
        for pos in 0..t {
            for kv in 0..N_KV {
                let o = (pos * N_KV + kv) * 132;
                let cnorm = 0.5 + 3.5 * unit(&mut rnd);
                kh[o..o + 4].copy_from_slice(&cnorm.to_le_bytes());
                for i in 0..128 {
                    kh[o + 4 + i] = (rnd() & 0xFF) as u8;
                }
            }
        }
        // V q8: 8 blocks x [f16 scale in [2^-3, 2)][32 int8] per (pos, kv).
        let mut vq = vec![0u8; t * N_KV * 8 * 34];
        for pos in 0..t {
            for kv in 0..N_KV {
                for b in 0..8 {
                    let o = (pos * N_KV + kv) * 8 * 34 + b * 34;
                    let exp = 12 + (rnd() % 4) as u16; // f16 exponent 12..=15
                    let scale = (exp << 10) | (rnd() & 0x3FF) as u16;
                    vq[o..o + 2].copy_from_slice(&scale.to_le_bytes());
                    for i in 0..32 {
                        vq[o + 2 + i] = (rnd() & 0xFF) as u8;
                    }
                }
            }
        }
        // +-1 sign tables.
        let mut sign_table = || -> Vec<f32> {
            (0..256)
                .map(|_| if rnd() & 1 == 0 { 1.0 } else { -1.0 })
                .collect()
        };
        let signs1 = sign_table();
        let signs2 = sign_table();
        let pos = vec![(t - 1) as i32];

        let upload = |gpu: &mut Gpu, data: &[u8]| -> rdna_compute::GpuTensor {
            let t = gpu.alloc_tensor(&[data.len()], DType::Raw).expect("alloc");
            gpu.hip.memcpy_htod(&t.buf, data).expect("upload");
            t
        };
        let as_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let q = upload(&mut gpu, &as_bytes(&qh));
        let k = upload(&mut gpu, &kh);
        let v8 = upload(&mut gpu, &vq);
        let s1 = upload(&mut gpu, &as_bytes(&signs1));
        let s2 = upload(&mut gpu, &as_bytes(&signs2));
        let posb = upload(
            &mut gpu,
            &pos.iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let mt = (max_seq + 127) / 128;
        let out = upload(&mut gpu, &vec![0u8; N_HEADS * HD * 4]);

        let run = |gpu: &mut Gpu, gqa: bool| -> Vec<f32> {
            let partials = upload(gpu, &vec![0u8; N_HEADS * mt * (2 + HD) * 4]);
            if gqa {
                gpu.attention_flash_fwht4_tile_gqa_gfx11(
                    &q, &k, &v8, &out, &posb.buf, &s1, &s2, t, N_HEADS, N_KV, HD, max_seq,
                    &partials, 8,
                )
                .expect("gqa tile");
            } else {
                gpu.attention_flash_fwht4(
                    &q, &k, &v8, &out, &posb.buf, &s1, &s2, t, N_HEADS, N_KV, HD, max_seq,
                    &partials, 8,
                )
                .expect("scalar tile");
            }
            let mut host = vec![0f32; N_HEADS * mt * (2 + HD)];
            gpu.hip
                .memcpy_dtoh(
                    unsafe {
                        std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, host.len() * 4)
                    },
                    &partials.buf,
                )
                .expect("download");
            host
        };

        let a = run(&mut gpu, false);
        let b = run(&mut gpu, true);
        assert_eq!(a.len(), b.len());
        let per = 2 + HD;
        let mut first = None;
        let mut maxd = 0f32;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            maxd = maxd.max((x - y).abs());
            if first.is_none() && x.to_bits() != y.to_bits() {
                let h = i / (mt * per);
                let r = i % (mt * per);
                first = Some((h, r / per, r % per, *x, *y));
            }
        }
        assert!(
            first.is_none(),
            "T={t}: GQA partials differ from scalar: max_abs_diff={maxd:e} first_diff \
             (head, tile, dim, scalar, gqa)={first:?}"
        );
        eprintln!("T={t}: bit-identical ({} partials)", a.len());
    }
    eprintln!("ORACLE-DONE");
}
