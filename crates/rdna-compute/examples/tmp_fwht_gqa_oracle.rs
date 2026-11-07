// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! TEMPORARY diagnostic (delete before commit): fwht4 GQA tile vs scalar
//! tile differential on synthetic inputs. Same q/K/V/signs/positions into
//! `attention_flash_fwht4` (scalar) and `attention_flash_fwht4_tile_gqa_gfx11`
//! (GQA); compares the `[n_heads, max_tiles, 2+256]` partials bitwise and
//! reports the first differing (head, tile, dim), the max abs diff, and a
//! per-region breakdown (max/sum vs O rows). Run with `--features lab`.
//!
//! Run: `cargo run -p rdna-compute --features lab --example tmp_fwht_gqa_oracle`

use rdna_compute::{DType, Gpu};

const N_HEADS: usize = 24;
const N_KV: usize = 4;
const HD: usize = 256;
const MAX_SEQ: usize = 512;

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

    for T in [1usize, 2, 16, 17, 32, 128, 256, 66048] {
    eprintln!("=== T={T} ===");
    // Size the allocation cap to the case: the launcher needs
    // max_seq >= T and partials sized for max_tiles.
    let max_seq = ((T + 127) / 128 * 128).max(512);

    let mut s = 0x12345678u64;
    let mut rnd = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    let rnd_f32 = |r: &mut dyn FnMut() -> u32| (r() as f32 / u32::MAX as f32 - 0.5) * 2.0;

    // q: random f32.
    let mut qh = vec![0f32; N_HEADS * HD];
    for v in qh.iter_mut() {
        *v = rnd_f32(&mut rnd);
    }
    // K: fwht4 records [f32 cnorm=1][128 random bytes] per (pos, kv).
    let mut kh = vec![0u8; T * N_KV * 132];
    for pos in 0..T {
        for kv in 0..N_KV {
            let o = (pos * N_KV + kv) * 132;
            kh[o..o + 4].copy_from_slice(&1f32.to_le_bytes());
            for i in 0..128 {
                kh[o + 4 + i] = (rnd() & 0xFF) as u8;
            }
        }
    }
    // Localize aid: duplicate record 0 over record 1 (same bytes must score
    // the same; any tl>=1-only divergence is an offset bug, not data).
    if T >= 2 {
        for kv in 0..N_KV {
            let (s, d) = (kv * 132, (N_KV + kv) * 132);
            kh.copy_within(s..s + 132, d);
        }
        // Verify the duplicate actually landed (else the test is vacuous).
        for kv in 0..N_KV {
            assert_eq!(
                &kh[kv * 132..kv * 132 + 132],
                &kh[(N_KV + kv) * 132..(N_KV + kv) * 132 + 132],
                "dup failed kv={kv}"
            );
        }
        eprintln!("DUP-OK T={T}");
    }
    // Structural control: make EVERY key byte-identical (copy record 0 over
    // the whole cache). Any divergence then is structural (barriers, loop
    // state, indexing) — never data. Enable with ALL_IDENTICAL=1.
    if std::env::var("ALL_IDENTICAL").as_deref() == Ok("1") {
        for pos in 1..T {
            for kv in 0..N_KV {
                let (s, d) = (kv * 132, (pos * N_KV + kv) * 132);
                kh.copy_within(s..s + 132, d);
            }
        }
        eprintln!("ALL-IDENTICAL T={T}");
    }
    // Structural probe: zero-K run. Set ZERO_K=1 in the environment to run
    // with an all-zero K cache instead: every score is then exactly 0 and
    // every tile sum exactly 2.0 in both implementations. Any deviation is
    // structural (offsets, barriers, reduction), independent of dequant.
    if std::env::var("ZERO_K").as_deref() == Ok("1") {
        for b in kh.iter_mut() {
            *b = 0;
        }
    }
    // V lloyd4: same layout as K.
    let mut vh = kh.clone();
    // V q8: 8 blocks x [f16 scale=1][32 int8] per (pos, kv).
    let mut vq = vec![0u8; T * N_KV * 8 * 34];
    for pos in 0..T {
        for kv in 0..N_KV {
            for b in 0..8 {
                let o = (pos * N_KV + kv) * 8 * 34 + b * 34;
                vq[o..o + 2].copy_from_slice(&0x3C00u16.to_le_bytes()); // 1.0f16
                for i in 0..32 {
                    vq[o + 2 + i] = (rnd() & 0xFF) as u8 as i8 as u8;
                }
            }
        }
    }
    let signs = vec![1f32; 256];
    let pos = vec![(T - 1) as i32];

    let upload = |gpu: &mut Gpu, data: &[u8]| -> rdna_compute::GpuTensor {
        let t = gpu.alloc_tensor(&[data.len()], DType::Raw).expect("alloc");
        gpu.hip.memcpy_htod(&t.buf, data).expect("upload");
        t
    };
    let q = upload(&mut gpu, unsafe {
        std::slice::from_raw_parts(qh.as_ptr() as *const u8, qh.len() * 4)
    });
    let k = upload(&mut gpu, &kh);
    let v4 = upload(&mut gpu, &vh);
    let v8t = upload(&mut gpu, &vq);
    let s1 = upload(&mut gpu, unsafe {
        std::slice::from_raw_parts(signs.as_ptr() as *const u8, signs.len() * 4)
    });
    let s2 = upload(&mut gpu, unsafe {
        std::slice::from_raw_parts(signs.as_ptr() as *const u8, signs.len() * 4)
    });
    let posb = upload(&mut gpu, unsafe {
        std::slice::from_raw_parts(pos.as_ptr() as *const u8, pos.len() * 4)
    });
    let mt = (max_seq + 127) / 128;
    let out = upload(&mut gpu, &vec![0u8; N_HEADS * HD * 4]);

    let run = |gpu: &mut Gpu, v: &rdna_compute::GpuTensor, vm: i32, gqa: bool| -> Vec<f32> {
        let partials = upload(gpu, &vec![0u8; N_HEADS * mt * (2 + HD) * 4]);
        if gqa {
            gpu.attention_flash_fwht4_tile_gqa_gfx11(
                &q, &k, v, &out, &posb.buf, &s1, &s2, T, N_HEADS, N_KV, HD, max_seq,
                &partials, vm,
            )
            .expect("gqa tile");
        } else {
            gpu.attention_flash_fwht4(
                &q, &k, v, &out, &posb.buf, &s1, &s2, T, N_HEADS, N_KV, HD, max_seq,
                &partials, vm,
            )
            .expect("scalar tile");
        }
        let mut host = vec![0f32; N_HEADS * mt * (2 + HD)];
        gpu.hip
            .memcpy_dtoh(
                unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr() as *mut u8, host.len() * 4) },
                &partials.buf,
            )
            .expect("download");
        host
    };

    for (v, vm, name) in [(&v4, 4, "lloyd4"), (&v8t, 8, "q8")] {
        let a = run(&mut gpu, v, vm, false);
        let b = run(&mut gpu, v, vm, true);
        assert_eq!(a.len(), b.len());
        let mut first = None;
        let mut maxd = 0f32;
        let mut maxi = 0;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let d = (x - y).abs();
            if d > maxd {
                maxd = d;
                maxi = i;
            }
            if first.is_none() && x.to_bits() != y.to_bits() {
                let per = 2 + HD;
                let h = i / (mt * per);
                let r = i % (mt * per);
                first = Some((h, r / per, r % per));
            }
        }
        eprintln!(
            "V={name}: len={} max_abs_diff={maxd:e} at flat {maxi} first_diff={first:?}",
            a.len()
        );
        let mut shown = 0;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if x.to_bits() != y.to_bits() && shown < 4 {
                eprintln!("  flat {i}: scalar={x:e} gqa={y:e}");
                shown += 1;
            }
        }
    }
    } // end T loop
    eprintln!("ORACLE-DONE");
}
