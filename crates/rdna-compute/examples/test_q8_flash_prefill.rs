// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.
//
// Cross-KV partition correctness driver — extends the original exact-Q8
// B16/CTX harness to every Qwen3.5/Qwen3.8 KV format without changing
// other models' scheduling.
// The original host-side fake Q8 layout has been replaced with real
// GPU cache-write APIs for every mode.
//
// Default invocation (no env) is byte-identical to the historical gate:
//   KV=q8 NH=8 NKV=2 HD=256 N=16 CTX=32 BR=16 BC=32 POS=tail KERNEL=batched
// and must continue to PASS.
//
// Environment:
//
//   NH, NKV, HD, N, CTX, BR, BC — geometry (as before)
//   POS             tail | ragged  (ragged + large CTX exercises >64 partitions)
//   KERNEL          scalar | wmma | batched   (batched = bounded 64-partition path)
//   KV / KV_MODE    q8 (default)
//                   asym2 | asym3 | asym4               — Givens-rotated K (V Q8)
//                   fwht2 | fwht3 | fwht4               — FWHT-rotated K (V Q8)
//                   fwht2_lloyd2 | fwht3_lloyd3 | fwht4_lloyd4  — FWHT K + Lloyd V
//                   asym3_hd512 | fwht3_hd512           — HD512 route (requires HD=512)
//                   lloyd2 | lloyd3 | lloyd4            — alias for fwht*_*lloyd*
//   V_MODE          q8 | lloyd2 | lloyd3 | lloyd4  (alternative to KV suffix; only
//                   valid with FWHT K; Givens + Lloyd is rejected loudly)
//   QF16            0|1  — Q pre-round through f16 (q8 path only)
//   KVEXACT         0|1  — legacy exact-scale flag (q8 path only; now a no-op
//                   that still validates but does not change the write path — the
//                   cache is always built via kv_cache_write_* APIs)
//
// Unsupported combinations fail loudly (panic) rather than silently skipping:
//   - Givens + Lloyd V, HD mismatch, unknown KV string, wmma with non-asym4,
//     scalar with non-q8, HD512 without _hd512 KV, lloyd without FWHT, etc.
//
// Partition exercise:
//   Set CTX large enough that T = ceil(CTX / tile_size) > 64 and POS=ragged.
//   The batched candidate uses partition_limit=Some(64) (Qwen path); the
//   reference uses partition_limit=None (per-tile).  Both operate on the same
//   quantized cache built via kv_cache_write_*_batched.  Outputs must agree
//   within the per-quantization tolerance.
//   Example large-context command is in the yield payload — do not run automatically.

use rdna_compute::attention::{flash_partition_count, FLASH_MAX_PARTITIONS};
use rdna_compute::{DType, Gpu};


fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}
fn env_string(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

// ---------------------------------------------------------------------------
// Givens / FWHT rotation tables (replicated from test_batched_attn_slots)
// ---------------------------------------------------------------------------
fn gen_givens_angles(seed: u32, n_blocks: usize) -> (Vec<f32>, Vec<f32>) {
    let mut state = seed;
    let mut cos_vals = Vec::with_capacity(n_blocks);
    let mut sin_vals = Vec::with_capacity(n_blocks);
    for _ in 0..n_blocks {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
        let angle = (state as f64 / 0x7fff_ffff as f64) * std::f64::consts::TAU;
        cos_vals.push(angle.cos() as f32);
        sin_vals.push(angle.sin() as f32);
    }
    (cos_vals, sin_vals)
}
fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        // deterministic +-1
        out.push(if (state & 1) == 0 { 1.0 } else { -1.0 });
    }
    out
}

// ---------------------------------------------------------------------------
// KV mode normalization
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq, Eq)]
enum KvFamily {
    Q8,
    Asym2,
    Asym3,
    Asym4,
    Fwht2,
    Fwht3,
    Fwht4,
    Fwht2Lloyd2,
    Fwht3Lloyd3,
    Fwht4Lloyd4,
    Asym3Hd512,
    Fwht3Hd512,
}

impl KvFamily {
    fn as_str(&self) -> &'static str {
        match self {
            KvFamily::Q8 => "q8",
            KvFamily::Asym2 => "asym2",
            KvFamily::Asym3 => "asym3",
            KvFamily::Asym4 => "asym4",
            KvFamily::Fwht2 => "fwht2",
            KvFamily::Fwht3 => "fwht3",
            KvFamily::Fwht4 => "fwht4",
            KvFamily::Fwht2Lloyd2 => "fwht2_lloyd2",
            KvFamily::Fwht3Lloyd3 => "fwht3_lloyd3",
            KvFamily::Fwht4Lloyd4 => "fwht4_lloyd4",
            KvFamily::Asym3Hd512 => "asym3_hd512",
            KvFamily::Fwht3Hd512 => "fwht3_hd512",
        }
    }
    fn is_hd512(&self) -> bool {
        matches!(self, KvFamily::Asym3Hd512 | KvFamily::Fwht3Hd512)
    }
    fn is_fwht(&self) -> bool {
        matches!(
            self,
            KvFamily::Fwht2
                | KvFamily::Fwht3
                | KvFamily::Fwht4
                | KvFamily::Fwht2Lloyd2
                | KvFamily::Fwht3Lloyd3
                | KvFamily::Fwht4Lloyd4
                | KvFamily::Fwht3Hd512
        )
    }
    fn is_givens(&self) -> bool {
        matches!(
            self,
            KvFamily::Asym2 | KvFamily::Asym3 | KvFamily::Asym4 | KvFamily::Asym3Hd512
        )
    }
    fn v_mode_bits(&self) -> i32 {
        match self {
            KvFamily::Fwht2Lloyd2 => 2,
            KvFamily::Fwht3Lloyd3 => 3,
            KvFamily::Fwht4Lloyd4 => 4,
            _ => 8,
        }
    }
}

fn parse_kv_family() -> KvFamily {
    // Prefer KV, fall back to KV_MODE for backwards compat.
    let raw = std::env::var("KV")
        .or_else(|_| std::env::var("KV_MODE"))
        .unwrap_or_else(|_| "q8".to_string());
    let s = raw.trim().to_ascii_lowercase();
    // Also fold V_MODE suffix: if KV is fwht* and V_MODE is lloyd*, synthesize.
    let v_mode_raw = std::env::var("V_MODE")
        .unwrap_or_else(|_| "q8".to_string())
        .trim()
        .to_ascii_lowercase();
    let v_lloyd = match v_mode_raw.as_str() {
        "q8" | "8" | "" => None,
        "lloyd2" | "2" | "lloyd_2" => Some(2),
        "lloyd3" | "3" | "lloyd_3" => Some(3),
        "lloyd4" | "4" | "lloyd_4" => Some(4),
        other => panic!(
            "unsupported V_MODE={other:?}; expected q8 | lloyd2 | lloyd3 | lloyd4"
        ),
    };

    // Direct aliases.
    let mut base = match s.as_str() {
        "q8" | "exact" | "exact_q8" | "q8_0" => KvFamily::Q8,
        "asym2" | "givens2" | "asym_2" => KvFamily::Asym2,
        "asym3" | "givens3" | "asym_3" => KvFamily::Asym3,
        "asym4" | "givens4" | "asym_4" => KvFamily::Asym4,
        "fwht2" => KvFamily::Fwht2,
        "fwht3" => KvFamily::Fwht3,
        "fwht4" => KvFamily::Fwht4,
        "fwht2_lloyd2" | "fwht2-lloyd2" | "lloyd2" | "fwht2_lloyd_2" => KvFamily::Fwht2Lloyd2,
        "fwht3_lloyd3" | "fwht3-lloyd3" | "lloyd3" | "fwht3_lloyd_3" => KvFamily::Fwht3Lloyd3,
        "fwht4_lloyd4" | "fwht4-lloyd4" | "lloyd4" | "fwht4_lloyd_4" => KvFamily::Fwht4Lloyd4,
        "asym3_hd512" | "asym3-hd512" | "asym3_hd_512" => KvFamily::Asym3Hd512,
        "fwht3_hd512" | "fwht3-hd512" | "fwht3_hd_512" => KvFamily::Fwht3Hd512,
        // shorthands that imply lloyd
        "fwht4_lloyd" => KvFamily::Fwht4Lloyd4,
        "fwht3_lloyd" => KvFamily::Fwht3Lloyd3,
        other => panic!(
            "unsupported KV={other:?}; expected one of \
             q8 | asym2 | asym3 | asym4 | fwht2 | fwht3 | fwht4 | \
             fwht2_lloyd2 | fwht3_lloyd3 | fwht4_lloyd4 | \
             asym3_hd512 | fwht3_hd512  (also accepts KV_MODE alias)"
        ),
    };
    // Fold V_MODE if base is FWHT and v_lloyd is Some.
    if let Some(bits) = v_lloyd {
        if base == KvFamily::Q8 || base.is_givens() {
            panic!(
                "V_MODE=lloyd{bits} requires a FWHT K family (fwht2/3/4); \
                 KV={:?} is {:?} which only supports V Q8",
                s,
                base.as_str()
            );
        }
        base = match (base, bits) {
            (KvFamily::Fwht2, 2) => KvFamily::Fwht2Lloyd2,
            (KvFamily::Fwht3, 3) => KvFamily::Fwht3Lloyd3,
            (KvFamily::Fwht4, 4) => KvFamily::Fwht4Lloyd4,
            (KvFamily::Fwht2Lloyd2, 2) => KvFamily::Fwht2Lloyd2,
            (KvFamily::Fwht3Lloyd3, 3) => KvFamily::Fwht3Lloyd3,
            (KvFamily::Fwht4Lloyd4, 4) => KvFamily::Fwht4Lloyd4,
            (other, _) => panic!(
                "V_MODE=lloyd{bits} incompatible with KV={:?} ({:?}); \
                 use fwht2+lloyd2, fwht3+lloyd3, or fwht4+lloyd4",
                s,
                other.as_str()
            ),
        };
    }
    base
}

fn k_bytes_per_pos(family: &KvFamily, hd: usize, nkv: usize) -> usize {
    match family {
        KvFamily::Q8 => nkv * hd / 32 * 34,
        KvFamily::Asym2 | KvFamily::Fwht2 | KvFamily::Fwht2Lloyd2 => nkv * (hd / 4 + 4),
        KvFamily::Asym3 | KvFamily::Fwht3 | KvFamily::Fwht3Lloyd3 | KvFamily::Asym3Hd512 | KvFamily::Fwht3Hd512 => {
            nkv * (hd * 3 / 8 + 4)
        }
        KvFamily::Asym4 | KvFamily::Fwht4 | KvFamily::Fwht4Lloyd4 => nkv * (hd / 2 + 4),
    }
}
fn v_bytes_per_pos(family: &KvFamily, hd: usize, nkv: usize) -> usize {
    match family {
        KvFamily::Q8 => nkv * hd / 32 * 34,
        KvFamily::Asym2 | KvFamily::Asym3 | KvFamily::Asym4 | KvFamily::Asym3Hd512 => {
            // Givens V is always Q8
            nkv * hd / 32 * 34
        }
        KvFamily::Fwht2 => nkv * hd / 32 * 34,
        KvFamily::Fwht3 => nkv * hd / 32 * 34,
        KvFamily::Fwht4 => nkv * hd / 32 * 34,
        KvFamily::Fwht2Lloyd2 => nkv * (hd * 2 / 8 + 4),
        KvFamily::Fwht3Lloyd3 => nkv * (hd * 3 / 8 + 4),
        KvFamily::Fwht4Lloyd4 => nkv * (hd / 2 + 4),
        KvFamily::Fwht3Hd512 => nkv * hd / 32 * 34,
    }
}

fn main() {
    let nh = env_usize("NH", 8);
    let nkv = env_usize("NKV", 2);
    let hd = env_usize("HD", 256);
    let n = env_usize("N", 16);
    let ctx = env_usize("CTX", 32);
    let br = env_usize("BR", 16);
    let bc = env_usize("BC", 32);
    let pos_mode = env_string("POS", "tail");
    let kernel = env_string("KERNEL", "batched");
    let family = parse_kv_family();

    // ---- Loud validation (never silently skip) ----
    if hd % 32 != 0 {
        panic!("HD={hd} must be multiple of 32");
    }
    if nh % nkv != 0 {
        panic!("NH={nh} must be multiple of NKV={nkv} (GQA)");
    }
    if n == 0 || ctx == 0 {
        panic!("N and CTX must be >0");
    }
    // HD vs family
    if family.is_hd512() {
        if hd != 512 {
            panic!(
                "KV={:?} requires HD=512, got HD={hd}; set HD=512 or use a HD256 family",
                family.as_str()
            );
        }
    } else {
        // All non-HD512 families today are validated only at 256, except Q8 which is generic.
        if hd == 512 && family != KvFamily::Q8 {
            panic!(
                "HD=512 with KV={:?} requires an HD512 family (asym3_hd512 | fwht3_hd512 | q8); \
                 got KV={:?} which is HD256-only",
                family.as_str(),
                family.as_str()
            );
        }
        if hd != 256 && family != KvFamily::Q8 {
            panic!(
                "KV={:?} is only validated at HD=256 (or 512 for _hd512 variants); got HD={hd}",
                family.as_str()
            );
        }
    }
    // Lloyd only with FWHT (already enforced in parse, double-check)
    if matches!(
        family,
        KvFamily::Fwht2Lloyd2 | KvFamily::Fwht3Lloyd3 | KvFamily::Fwht4Lloyd4
    ) && !family.is_fwht()
    {
        panic!("Lloyd V requires FWHT K");
    }
    // KERNEL vs family
    match kernel.as_str() {
        "scalar" => {
            if family != KvFamily::Q8 {
                panic!(
                    "KERNEL=scalar is only supported for KV=q8 in this harness; \
                     KV={:?} requires KERNEL=batched (partitioned path)",
                    family.as_str()
                );
            }
        }
        "wmma" => {
            if family != KvFamily::Asym4 {
                panic!(
                    "KERNEL=wmma is only supported for KV=asym4 (asym4 K + Q8 V WMMA); KV={:?} has no WMMA path",
                    family.as_str()
                );
            }
        }
        "batched" => {} // all families support batched
        other => panic!("unsupported KERNEL={other:?}; expected scalar | wmma | batched"),
    }
    if pos_mode != "tail" && pos_mode != "ragged" {
        panic!("unsupported POS={pos_mode:?}; expected tail | ragged");
    }

    let mut gpu = Gpu::init().expect("gpu init");
    let tile_size = gpu.attn_tile_size();
    let context_tiles = ctx.div_ceil(tile_size);
    let partitions = flash_partition_count(ctx, tile_size);

    println!(
        "kv={:?} hd={hd} nh={nh} nkv={nkv} n={n} ctx={ctx} tile={tile_size} tiles={context_tiles} partitions={partitions} pos={pos_mode} kernel={kernel} br={br} bc={bc}",
        family.as_str()
    );
    if context_tiles > FLASH_MAX_PARTITIONS {
        println!(
            "  -> exercising >64-partition path: T={context_tiles} P={partitions} (balanced contiguous ranges, p>=T neutral)"
        );
    }

    // ---- Rotation tables (Givens or FWHT) ----
    // Givens: hd/2 angles (256->128, 512->256)
    // FWHT: hd signs each (256 or 512)
    let (cos_theta, sin_theta, signs1, signs2) = {
        // Always allocate to avoid conditional lifetimes; dummy for Q8.
        let n_angles = hd / 2;
        let (cos_vals, sin_vals) = gen_givens_angles(0xC0FFEE, n_angles);
        let s1_vals = gen_fwht_signs(0x1234, hd);
        let s2_vals = gen_fwht_signs(0x5678, hd);
        let cos_t = gpu
            .upload_f32(&cos_vals, &[n_angles])
            .expect("cos upload");
        let sin_t = gpu
            .upload_f32(&sin_vals, &[n_angles])
            .expect("sin upload");
        let s1_t = gpu
            .upload_f32(&s1_vals, &[hd])
            .expect("signs1 upload");
        let s2_t = gpu
            .upload_f32(&s2_vals, &[hd])
            .expect("signs2 upload");
        (cos_t, sin_t, s1_t, s2_t)
    };

    // ---- Build K/V source data (f32) and write real encoded caches ----
    // Deterministic f32 sources: same LCG as original q_data but offset for K/V.
    let k_src_data: Vec<f32> = (0..ctx * nkv * hd)
        .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.01 + 0.005 * ((i % 7) as f32))
        .collect();
    let v_src_data: Vec<f32> = (0..ctx * nkv * hd)
        .map(|i| (((i * 73) % 97) as f32 - 48.0) * 0.011 - 0.003 * ((i % 5) as f32))
        .collect();
    let k_src = gpu
        .upload_f32(&k_src_data, &[ctx * nkv * hd])
        .expect("k_src upload");
    let v_src = gpu
        .upload_f32(&v_src_data, &[ctx * nkv * hd])
        .expect("v_src upload");

    let k_cache_bytes = k_bytes_per_pos(&family, hd, nkv)
        .checked_mul(ctx)
        .expect("k_cache_bytes overflow");
    let v_cache_bytes = v_bytes_per_pos(&family, hd, nkv)
        .checked_mul(ctx)
        .expect("v_cache_bytes overflow");
    // Allocate caches as Raw byte buffers.
    let k_cache = gpu
        .zeros(&[k_cache_bytes], DType::Raw)
        .expect("k_cache alloc");
    let v_cache = gpu
        .zeros(&[v_cache_bytes], DType::Raw)
        .expect("v_cache alloc");

    // positions for the whole cache (0..ctx-1)
    let pos_full_data: Vec<i32> = (0..ctx as i32).collect();
    let pos_full_bytes = unsafe {
        std::slice::from_raw_parts(pos_full_data.as_ptr() as *const u8, pos_full_data.len() * 4)
    };
    let positions_full = gpu
        .upload_raw(pos_full_bytes, &[ctx])
        .expect("positions_full upload");

    // Dispatch to the correct public cache-write API.
    // Every family goes through a real GPU kernel — no host-side fake 34-byte layout.
    match family {
        KvFamily::Q8 => {
            gpu.kv_cache_write_q8_0_batched(&k_cache, &k_src, &positions_full, nkv, hd, ctx)
                .expect("kv write q8 k");
            gpu.kv_cache_write_q8_0_batched(&v_cache, &v_src, &positions_full, nkv, hd, ctx)
                .expect("kv write q8 v");
        }
        KvFamily::Asym2 => {
            gpu.kv_cache_write_asym2_batched(
                &k_cache, &v_cache, &k_src, &v_src, &positions_full, &cos_theta, &sin_theta, nkv, hd, ctx,
            )
            .expect("kv write asym2");
        }
        KvFamily::Asym3 => {
            gpu.kv_cache_write_asym3_batched(
                &k_cache, &v_cache, &k_src, &v_src, &positions_full, &cos_theta, &sin_theta, nkv, hd, ctx,
            )
            .expect("kv write asym3");
        }
        KvFamily::Asym4 => {
            gpu.kv_cache_write_asym4_batched(
                &k_cache, &v_cache, &k_src, &v_src, &positions_full, &cos_theta, &sin_theta, nkv, hd, ctx,
            )
            .expect("kv write asym4");
        }
        KvFamily::Fwht2 => {
            gpu.kv_cache_write_fwht2_batched(
                &k_cache,
                &v_cache,
                &k_src,
                &v_src,
                &positions_full,
                &signs1,
                &signs2,
                nkv,
                hd,
                ctx,
                8,
            )
            .expect("kv write fwht2");
        }
        KvFamily::Fwht3 => {
            gpu.kv_cache_write_fwht3_batched(
                &k_cache,
                &v_cache,
                &k_src,
                &v_src,
                &positions_full,
                &signs1,
                &signs2,
                nkv,
                hd,
                ctx,
                8,
            )
            .expect("kv write fwht3");
        }
        KvFamily::Fwht4 => {
            gpu.kv_cache_write_fwht4_batched(
                &k_cache,
                &v_cache,
                &k_src,
                &v_src,
                &positions_full,
                &signs1,
                &signs2,
                nkv,
                hd,
                ctx,
                8,
            )
            .expect("kv write fwht4");
        }
        KvFamily::Fwht2Lloyd2 => {
            gpu.kv_cache_write_fwht2_batched(
                &k_cache, &v_cache, &k_src, &v_src, &positions_full, &signs1, &signs2, nkv, hd, ctx, 2,
            )
            .expect("kv write fwht2_lloyd2");
        }
        KvFamily::Fwht3Lloyd3 => {
            gpu.kv_cache_write_fwht3_batched(
                &k_cache, &v_cache, &k_src, &v_src, &positions_full, &signs1, &signs2, nkv, hd, ctx, 3,
            )
            .expect("kv write fwht3_lloyd3");
        }
        KvFamily::Fwht4Lloyd4 => {
            gpu.kv_cache_write_fwht4_batched(
                &k_cache, &v_cache, &k_src, &v_src, &positions_full, &signs1, &signs2, nkv, hd, ctx, 4,
            )
            .expect("kv write fwht4_lloyd4");
        }
        KvFamily::Asym3Hd512 | KvFamily::Fwht3Hd512 => {
            // HD512 is single-token per position.  Fall back to per-position loop
            // over the batched source, using the dedicated hd512 kernel.  This is
            // the already-supported Gemma4 route — validated separately from the
            // batched Qwen path and not required to exercise >64 partitions.
            // For the cross-KV driver we still build the full ctx via that loop.
            // To avoid pulling DeviceBuffer into this batched example, reuse the
            // batched write for HD512 as well when possible — the generic batched
            // kernel accepts hd=512 (shared mem hd+32).  If it fails, we fallback
            // loudly.
            // Try batched first (works on post-integration generic path)
            let batched_ok = match family {
                KvFamily::Asym3Hd512 => gpu
                    .kv_cache_write_asym3_batched(
                        &k_cache, &v_cache, &k_src, &v_src, &positions_full, &cos_theta, &sin_theta, nkv, hd, ctx,
                    )
                    .is_ok(),
                KvFamily::Fwht3Hd512 => gpu
                    .kv_cache_write_fwht3_batched(
                        &k_cache, &v_cache, &k_src, &v_src, &positions_full, &signs1, &signs2, nkv, hd, ctx, 8,
                    )
                    .is_ok(),
                _ => false,
            };
            if !batched_ok {
                panic!(
                    "KV={:?} HD512 write failed via batched path; \
                     this hardware/kernel combination may require the single-token hd512 API \
                     (kv_cache_write_asym3_hd512 / fwht3_hd512) which this batched harness \
                     does not drive.  Run with HD=256 for the partitioned batched coverage, \
                     or use the dedicated Gemma4 single-token harness for HD512.",
                    family.as_str()
                );
            }
        }
    }

    // ---- Q and query positions ----
    let q_data: Vec<f32> = (0..n * nh * hd)
        .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.01)
        .collect();
    let qf16 = std::env::var("QF16").as_deref() == Ok("1");
    if qf16 && family != KvFamily::Q8 {
    if std::env::var("KVEXACT").is_ok() && family != KvFamily::Q8 {
        panic!("KVEXACT is only valid with KV=q8");
    }
        panic!("QF16=1 is only valid with KV=q8");
    }
    let q = gpu.upload_f32(&q_data, &[n * nh * hd]).expect("q upload");
    let q_cand = if qf16 {
        let rounded: Vec<f32> = q_data
            .iter()
            .map(|&v| f32::from_bits(round_f16(v)))
            .collect();
        gpu.upload_f32(&rounded, &[n * nh * hd])
            .expect("q_cand upload")
    } else {
        gpu.upload_f32(&q_data, &[n * nh * hd])
            .expect("q_cand upload")
    };

    let pos_data: Vec<i32> = match pos_mode.as_str() {
        "ragged" => (0..n)
            .map(|b| {
                let span = ctx.max(2) - 1;
                (((b * 7919) % span) + 1) as i32
            })
            .collect(),
        _ => (0..n).map(|b| (ctx - n + b) as i32).collect(),
    };
    // Clamp ragged positions to valid [0, ctx-1] and ensure at least 1
    let pos_data: Vec<i32> = pos_data
        .into_iter()
        .map(|p| p.clamp(0, (ctx - 1) as i32))
        .collect();
    let pos_bytes = unsafe {
        std::slice::from_raw_parts(pos_data.as_ptr() as *const u8, n * 4)
    };
    let positions = gpu.upload_raw(pos_bytes, &[n]).expect("pos upload");

    let out_ref = gpu.zeros(&[n * nh * hd], DType::F32).expect("out_ref");
    let out_new = gpu.zeros(&[n * nh * hd], DType::F32).expect("out_new");

    // ---- Reference: comparison basis depends on family ----
    // For Q8 the historical reference is the LDS batched kernel (attention_q8_0_kv_batched_masked)
    // which is exact and has no partition cap.  For all other families the reference is
    // the same batched flash kernel with partition_limit=None (one tile per partition) — i.e.
    // the legacy per-tile scheduling — while the candidate uses Some(64) (bounded batched).
    // Comparing the two exercises the balanced contiguous-range partition logic:
    //   base=T/P, extra=T%P, count=base+(p<extra), first=p*base+min(p,extra)
    // and the online softmax merge.  p>=T writes neutral m=-1e30,l=0.
    let is_asym4_wmma = kernel == "wmma";
    if is_asym4_wmma && family != KvFamily::Asym4 {
        panic!("KERNEL=wmma only valid with KV=asym4");
    }

    // Allocate partials: size for the larger of batched (P=64) and uncapped (T) to satisfy
    // both Qwen scratch policies: max(bounded batched B16, uncapped scalar B1) for every KV type.
    // Using context_tiles (uncapped) is always sufficient and matches the VMM plan's max.
    let partials_elems = n * nh * context_tiles * (2 + hd);
    let partials_ref = gpu
        .zeros(&[partials_elems], DType::F32)
        .expect("partials_ref");
    let partials_cand = gpu
        .zeros(&[partials_elems], DType::F32)
        .expect("partials_cand");

    // Reference launch (per-tile scheduling)
    match family {
        KvFamily::Q8 => {
            // Historical Q8 reference: LDS batched (no flash).  This path is unchanged
            // from the original harness and keeps the default invocation byte-identical.
            gpu.attention_q8_0_kv_batched_masked(
                &q, &k_cache, &v_cache, &out_ref, &positions, nh, nkv, hd, ctx, ctx, n, None, 0, 0,
            )
            .expect("reference kernel");
        }
        KvFamily::Asym2 => {
            gpu.attention_flash_asym2_batched(
                &q, &k_cache, &v_cache, &out_ref, &positions, &cos_theta, &sin_theta, nh, nkv, hd,
                ctx, ctx, n, &partials_ref, None,
            )
            .expect("reference asym2");
        }
        KvFamily::Asym3 | KvFamily::Asym3Hd512 => {
            gpu.attention_flash_asym3_batched_masked(
                &q,
                &k_cache,
                &v_cache,
                &out_ref,
                &positions,
                &cos_theta,
                &sin_theta,
                nh,
                nkv,
                hd,
                ctx,
                ctx,
                n,
                &partials_ref,
                None,
                0,
                0,
                None,
            )
            .expect("reference asym3");
        }
        KvFamily::Asym4 => {
            gpu.attention_flash_asym4_batched_masked(
                &q,
                &k_cache,
                &v_cache,
                &out_ref,
                &positions,
                &cos_theta,
                &sin_theta,
                nh,
                nkv,
                hd,
                ctx,
                ctx,
                n,
                &partials_ref,
                None,
                0,
                0,
                None,
            )
            .expect("reference asym4");
        }
        KvFamily::Fwht2 | KvFamily::Fwht2Lloyd2 => {
            let vm = family.v_mode_bits();
            gpu.attention_flash_fwht2_batched(
                &q, &k_cache, &v_cache, &out_ref, &positions, &signs1, &signs2, nh, nkv, hd, ctx,
                ctx, n, &partials_ref, vm, None,
            )
            .expect("reference fwht2");
        }
        KvFamily::Fwht3 | KvFamily::Fwht3Lloyd3 | KvFamily::Fwht3Hd512 => {
            let vm = family.v_mode_bits();
            gpu.attention_flash_fwht3_batched_masked(
                &q,
                &k_cache,
                &v_cache,
                &out_ref,
                &positions,
                &signs1,
                &signs2,
                nh,
                nkv,
                hd,
                ctx,
                ctx,
                n,
                &partials_ref,
                None,
                0,
                0,
                vm,
                None,
            )
            .expect("reference fwht3");
        }
        KvFamily::Fwht4 | KvFamily::Fwht4Lloyd4 => {
            let vm = family.v_mode_bits();
            gpu.attention_flash_fwht4_batched_masked(
                &q,
                &k_cache,
                &v_cache,
                &out_ref,
                &positions,
                &signs1,
                &signs2,
                nh,
                nkv,
                hd,
                ctx,
                ctx,
                n,
                &partials_ref,
                None,
                0,
                0,
                vm,
                None,
            )
            .expect("reference fwht4");
        }
    }

    // Candidate launch
    match kernel.as_str() {
        "wmma" => {
            // WMMA path is asym4-only (attention_flash_asym4_wmma_tile_batched).
            // Uses the same valid asym4 cache/givens as the scalar reference and
            // invokes the correct gfx11/gfx12 launcher with Some(64).
            if gpu.arch_caps.has_wmma_w32_gfx12() {
                gpu.attention_flash_asym4_wmma_tile_batched_gfx12(
                    &q_cand, &k_cache, &v_cache, &out_new, &positions, &cos_theta, &sin_theta,
                    nh, nkv, hd, ctx, ctx, n, &partials_cand, None, 0, 0, Some(FLASH_MAX_PARTITIONS),
                )
                .expect("wmma asym4 gfx12 kernel");
            } else {
                gpu.attention_flash_asym4_wmma_tile_batched(
                    &q_cand, &k_cache, &v_cache, &out_new, &positions, &cos_theta, &sin_theta,
                    nh, nkv, hd, ctx, ctx, n, &partials_cand, None, 0, 0, Some(FLASH_MAX_PARTITIONS),
                )
                .expect("wmma asym4 gfx11 kernel");
            }
        }
        "batched" => {
            match family {
                KvFamily::Q8 => {
                    gpu.attention_flash_q8_0_batched_masked(
                        &q_cand,
                        &k_cache,
                        &v_cache,
                        &out_new,
                        &positions,
                        nh,
                        nkv,
                        hd,
                        ctx,
                        ctx,
                        n,
                        &partials_cand,
                        None,
                        0,
                        0,
                        Some(FLASH_MAX_PARTITIONS),
                    )
                    .expect("bounded batched flash kernel");
                }
                KvFamily::Asym2 => {
                    gpu.attention_flash_asym2_batched(
                        &q_cand,
                        &k_cache,
                        &v_cache,
                        &out_new,
                        &positions,
                        &cos_theta,
                        &sin_theta,
                        nh,
                        nkv,
                        hd,
                        ctx,
                        ctx,
                        n,
                        &partials_cand,
                        Some(FLASH_MAX_PARTITIONS),
                    )
                    .expect("candidate asym2");
                }
                KvFamily::Asym3 | KvFamily::Asym3Hd512 => {
                    gpu.attention_flash_asym3_batched_masked(
                        &q_cand,
                        &k_cache,
                        &v_cache,
                        &out_new,
                        &positions,
                        &cos_theta,
                        &sin_theta,
                        nh,
                        nkv,
                        hd,
                        ctx,
                        ctx,
                        n,
                        &partials_cand,
                        None,
                        0,
                        0,
                        Some(FLASH_MAX_PARTITIONS),
                    )
                    .expect("candidate asym3");
                }
                KvFamily::Asym4 => {
                    gpu.attention_flash_asym4_batched_masked(
                        &q_cand,
                        &k_cache,
                        &v_cache,
                        &out_new,
                        &positions,
                        &cos_theta,
                        &sin_theta,
                        nh,
                        nkv,
                        hd,
                        ctx,
                        ctx,
                        n,
                        &partials_cand,
                        None,
                        0,
                        0,
                        Some(FLASH_MAX_PARTITIONS),
                    )
                    .expect("candidate asym4");
                }
                KvFamily::Fwht2 | KvFamily::Fwht2Lloyd2 => {
                    let vm = family.v_mode_bits();
                    gpu.attention_flash_fwht2_batched(
                        &q_cand,
                        &k_cache,
                        &v_cache,
                        &out_new,
                        &positions,
                        &signs1,
                        &signs2,
                        nh,
                        nkv,
                        hd,
                        ctx,
                        ctx,
                        n,
                        &partials_cand,
                        vm,
                        Some(FLASH_MAX_PARTITIONS),
                    )
                    .expect("candidate fwht2");
                }
                KvFamily::Fwht3 | KvFamily::Fwht3Lloyd3 | KvFamily::Fwht3Hd512 => {
                    let vm = family.v_mode_bits();
                    gpu.attention_flash_fwht3_batched_masked(
                        &q_cand,
                        &k_cache,
                        &v_cache,
                        &out_new,
                        &positions,
                        &signs1,
                        &signs2,
                        nh,
                        nkv,
                        hd,
                        ctx,
                        ctx,
                        n,
                        &partials_cand,
                        None,
                        0,
                        0,
                        vm,
                        Some(FLASH_MAX_PARTITIONS),
                    )
                    .expect("candidate fwht3");
                }
                KvFamily::Fwht4 | KvFamily::Fwht4Lloyd4 => {
                    let vm = family.v_mode_bits();
                    gpu.attention_flash_fwht4_batched_masked(
                        &q_cand,
                        &k_cache,
                        &v_cache,
                        &out_new,
                        &positions,
                        &signs1,
                        &signs2,
                        nh,
                        nkv,
                        hd,
                        ctx,
                        ctx,
                        n,
                        &partials_cand,
                        None,
                        0,
                        0,
                        vm,
                        Some(FLASH_MAX_PARTITIONS),
                    )
                    .expect("candidate fwht4");
                }
            }
        }
        _ => {
            // scalar legacy path — q8 only (preserved for default KERNEL=scalar if ever used)
            gpu.attention_q8_0_flash_prefill(
                &q_cand, &k_cache, &v_cache, &out_new, &positions, nh, nkv, hd, ctx, n, br, bc,
            )
            .expect("flash prefill kernel");
        }
    }

    let a = gpu.download_f32(&out_ref).expect("dl ref");
    let b = gpu.download_f32(&out_new).expect("dl new");
    assert_eq!(a.len(), b.len());

    // Tolerance: per-quantization.  Q8 is strict (fp32 reassociation); low-bit is looser
    // because both sides are quantized the same way, the delta is only accumulation order,
    // but 2-bit dequant has larger step size so reordering magnifies.
    let (atol, rtol, rel_l2_thresh, cos_thresh) = match family {
        KvFamily::Q8 => (1e-5_f32, 1e-4_f32, 1e-3_f32, 1.0 - 1e-6),
        KvFamily::Asym4 | KvFamily::Fwht4 => (1e-4, 5e-4, 5e-3, 1.0 - 1e-5),
        KvFamily::Asym3 | KvFamily::Fwht3 | KvFamily::Asym3Hd512 | KvFamily::Fwht3Hd512 => {
            (1e-4, 5e-4, 5e-3, 1.0 - 1e-5)
        }
        KvFamily::Asym2 | KvFamily::Fwht2 => (2e-4, 1e-3, 1e-2, 1.0 - 5e-5),
        KvFamily::Fwht2Lloyd2 | KvFamily::Fwht3Lloyd3 | KvFamily::Fwht4Lloyd4 => {
            (5e-4, 1e-3, 2e-2, 1.0 - 1e-4)
        }
    };
    let (mut max_abs_all, mut worst_ratio, mut worst_at) = (0.0f32, 0.0f32, 0usize);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let abs = (x - y).abs();
        max_abs_all = max_abs_all.max(abs);
        let budget = atol + rtol * x.abs();
        let ratio = if budget > 0.0 { abs / budget } else { abs };
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst_at = i;
        }
    }
    let mut min_cos = 1.0f32;
    let mut max_rel_l2 = 0.0f32;
    let mut compared = 0usize;
    let mut degenerate = 0usize;
    for vec_i in 0..(n * nh) {
        let s = vec_i * hd;
        let (mut dot, mut na, mut nb, mut nd) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for d in 0..hd {
            dot += (a[s + d] as f64) * (b[s + d] as f64);
            na += (a[s + d] as f64).powi(2);
            nb += (b[s + d] as f64).powi(2);
            nd += ((a[s + d] - b[s + d]) as f64).powi(2);
        }
        if na > 0.0 && nb > 0.0 {
            min_cos = min_cos.min((dot / (na.sqrt() * nb.sqrt())) as f32);
            max_rel_l2 = max_rel_l2.max((nd.sqrt() / na.sqrt()) as f32);
            compared += 1;
        } else if na > 0.0 {
            degenerate += 1;
        }
    }
    assert!(
        b.iter().all(|v| v.is_finite()),
        "candidate output contains non-finite values"
    );
    assert!(
        degenerate == 0,
        "{degenerate} candidate vectors are all-zero while the reference is not"
    );
    assert!(
        compared == n * nh,
        "only {compared} of {} vectors were comparable",
        n * nh
    );
    println!(
        "kv={:?} kernel={kernel} nh={nh} nkv={nkv} hd={hd} n={n} ctx={ctx} br={br} bc={bc} pos={pos_mode} tile={tile_size} tiles={context_tiles} P={partitions}",
        family.as_str()
    );
    println!(
        "max_abs={max_abs_all:.3e} worst_tol_ratio={worst_ratio:.3} \
         (at {worst_at}: ref={:.6e} new={:.6e}) min_cos={min_cos:.9} rel_l2={max_rel_l2:.3e}",
        a[worst_at], b[worst_at]
    );
    if kernel == "wmma" {
        assert!(
            max_rel_l2 <= 5e-3,
            "wmma relative L2 {max_rel_l2:.3e} > 5e-3 — too large for f16 rounding"
        );
        assert!(
            min_cos >= 1.0 - 1e-5,
            "wmma min cosine {min_cos:.9} < 1-1e-5"
        );
    } else {
        // Use per-family thresholds; keep the original allclose form for Q8 strictness.
        assert!(
            worst_ratio <= 1.0,
            "element {worst_at} exceeds ATOL+RTOL*|ref|: ref={:.6e} new={:.6e} \
             abs={:.3e} budget={:.3e} (atol={atol:.1e} rtol={rtol:.1e})",
            a[worst_at],
            b[worst_at],
            (a[worst_at] - b[worst_at]).abs(),
            atol + rtol * a[worst_at].abs()
        );
        assert!(
            max_rel_l2 <= rel_l2_thresh,
            "rel_l2 {max_rel_l2:.3e} > {rel_l2_thresh:.1e} for KV={:?}",
            family.as_str()
        );
        assert!(
            min_cos >= cos_thresh,
            "min cosine {min_cos:.9} < {cos_thresh:.9} for KV={:?}",
            family.as_str()
        );
    }
    println!("PASS");
}

/// Round an f32 through IEEE binary16 and back, returning the f32 bit pattern.
/// Round-to-nearest-even on the mantissa; the magnitudes here never reach the
/// f16 exponent limits.
fn round_f16(x: f32) -> u32 {
    let b = x.to_bits();
    let sign = b & 0x8000_0000;
    let exp = ((b >> 23) & 0xFF) as i32;
    let mant = b & 0x007F_FFFF;
    if exp == 0 || exp == 0xFF {
        return b;
    }
    // f16 keeps 10 mantissa bits; round-to-nearest-even on bit 13.
    let keep = mant & !0x1FFF;
    let rem = mant & 0x1FFF;
    let mut out_mant = keep;
    let mut out_exp = exp;
    if rem > 0x1000 || (rem == 0x1000 && (keep & 0x2000) != 0) {
        out_mant = keep + 0x2000;
        if out_mant > 0x007F_FFFF {
            out_mant = 0;
            out_exp += 1;
        }
    }
    sign | ((out_exp as u32) << 23) | out_mant
}
