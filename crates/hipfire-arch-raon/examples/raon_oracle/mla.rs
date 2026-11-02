// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! raon_oracle `mla` module: runs the PRODUCTION latent-attention stages
//! (`execute_latent_attention`) on GPU against the reference fixtures
//! `<fixtures>/mla/<case>.safetensors` (+ optional shared `weights` file).
//!
//! Fixture tensors (F32 holding BF16-rounded values; mode "bf16" only):
//!   inputs : q_a_out[N,qlora] kv_a_out[N,latent+tail] q_a_norm[qlora]
//!            kv_a_norm[latent] q_proj[N,H*(nope+tail)] kv_b[H*(nope+v),latent]
//!            g_out[N,H*v] cache_prefix[P,latent+tail]
//!   outputs: qlat[N,qlora] cache_out[P+N,latent+tail]
//!            kv_expanded[P+N,H,nope+v] attn_y[N,H*v] gated[N,H*v]
//! dims: {N,P,H,nope,tail,v,latent,qlora}.
//!
//! Checks per case: qlat / cache rows / expanded K,V / attention / gate against
//! the reference (PLAN §6.2 BF16 band: rel_l2<=1e-2, cosine>=.9999, no
//! localized outlier); prefix rows bit-preserved; BITWISE equality of the
//! mirror built by two different decompression schedules and by the
//! gather -> external projector -> scatter route; repeated attention
//! byte-identical; cache poison beyond the live length never read.
//!
//! Attention routes (`--route reference|tiled|absorbed|all`, default all; the
//! reference route always runs first, it is the oracle):
//!   reference : per-head serial kernel over the expanded token-major mirror
//!               (`knope[t][h][nope]`, `vmir[t][h][v]`).
//!   tiled     : flash prefill (rows>1) / split-KV decode (rows==1) over the
//!               same mirror; BF16 band + bytewise repeat.
//!   absorbed  : opt-in absorbed-latent route, no mirror; NOT bit-exact to the
//!               decompressed routes, so every absorbed check is labelled
//!               `"route": "absorbed", "bit_exact": false`. It has no
//!               intermediate BF16 K/V rounding, so it is GATED (BF16 band)
//!               against an unrounded-f64 CPU reference computed from the
//!               fixture cache/kv_b/q. The comparison against the fixture
//!               `attn_y` (which carries the decompressed route's BF16 K/V
//!               rounding) is reported but non-gating, and `absorbed_vs_tiled`
//!               gates on rel_l2/cosine only (max_abs printed, not gating).
//!               Every check JSON carries `"gating": true|false`.
//!
//! Q8 mirror (`tiled_q8`, runs with the tiled route): the expanded K_nope/V
//! mirror may be Raw int8 planes (per (row, head, plane) F16 RNE scale
//! `min(max|x|/127, 65504)`, codes `clamp(rint(x/scale), ±127)`). Gated: the
//! External scatter bytes equal a host quantizer under two schedules (poison
//! untouched); tiled attention on a Q8 mirror built from the dense mirror is
//! byte-repeatable and within the BF16 band of the Reference route on the
//! dequantized mirror, of the fixture (short) and of the CPU reference (long).
//!
//! Generated long-context cases `long_p<P>_n<N>` (skipped with `--long 0` and
//! in `--negative` mode; selectable with `--case`): Raon geometry and the real
//! `weights_raon.safetensors` kv_b; the latent cache and the queries come from a
//! fixed-seed PRNG (xorshift64* + Box-Muller, seed [`LONG_SEED`] xor P<<20 xor N)
//! with a planted far-from-diagonal "needle"; the references are independent
//! multi-threaded CPU attentions (f64 scores/softmax) over f32-fma-expanded,
//! BF16-rounded K/V (decompressed routes) and over unrounded K/V (absorbed).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use hipfire_dispatch::pipeline::latent_attention::{
    execute_latent_attention, AbsorbOperands, AttentionRoute, ExpandedMirror, KvExpansion,
    LatentAttentionOp, LatentCache, LatentGeometry, RowRange,
};
use rdna_compute::latent_attention_absorb::latent_attention_absorb_packed_elems;
use rdna_compute::latent_attention_decode::latent_attention_decode_workspace_elems;
use rdna_compute::latent_attention_tiled::q8_plane_row_bytes;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};

const BF16_REL_L2: f64 = 1e-2;
const BF16_COSINE: f64 = 0.9999;
/// Max-abs gate as a fraction of max|reference| (catches localized corruption).
const BF16_MAX_ABS_FRAC: f64 = 2e-2;
/// BF16 bits of 1e30: finite, huge — any read of a poisoned row wrecks softmax.
const POISON_BF16: u16 = 0x7149;
const EXTERNAL_ROW_LIMIT: usize = 130;

/// `(P, N)` of the generated long-context cases.
const LONG_CASES: [(usize, usize); 7] = [
    (8191, 1),
    (8192, 1),
    (8192, 17),
    (16384, 1),
    (16384, 33),
    (27000, 1),
    (27000, 17),
];
/// Base seed of the generated cases (ASCII "MLARAON1"); per case it is xored
/// with `P << 20` and `N`, so every case is reproducible on its own.
const LONG_SEED: u64 = 0x4D4C_4152_414F_4E31;
/// Standard deviation of the generated queries (latent/tail rows are N(0,1)).
const LONG_QUERY_STD: f64 = 3.0;
/// Planted key tail = this * mean-over-heads tail of the last query row. Tuned
/// by scanning {2.6, 6, 10, 20, 40}: 2.6 never produced a planted peak above 0.3
/// softmax mass; 20 gives a far-range planted peak (mass > 0.3) to 24-27 of 40
/// heads at N=1 and to 27-32% of (row, head) tasks at N>1.
const LONG_NEEDLE_GAIN: f32 = 20.0;
const LONG_WEIGHTS: &str = "weights_raon.safetensors";
/// Decompress tile (rows per absolute-range launch) of the long cases.
const LONG_TILE: usize = 1024;
/// Raon-1-mini MLA geometry (matches the `weights_raon` manifest dims).
const RAON_GEOMETRY: LatentGeometry = LatentGeometry {
    n_heads: 40,
    nope: 128,
    tail: 64,
    v_dim: 128,
    latent: 512,
};

// ── safetensors (F32 / BF16 / F16 → f32) ────────────────────────────────────

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn f32_to_bf16(f: f32) -> u16 {
    let u = f.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        return ((u >> 16) | 0x40) as u16;
    }
    (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | ((e2 as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | (((e - 15 + 127) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

struct Tensors {
    map: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Tensors {
    fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        if bytes.len() < 8 {
            return Err(format!("{}: truncated safetensors", path.display()));
        }
        let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let hdr_end = 8usize
            .checked_add(hlen)
            .filter(|e| *e <= bytes.len())
            .ok_or_else(|| format!("{}: bad header length", path.display()))?;
        let header: Value = serde_json::from_slice(&bytes[8..hdr_end])
            .map_err(|e| format!("{}: header json: {e}", path.display()))?;
        let data = &bytes[hdr_end..];
        let mut map = HashMap::new();
        for (name, meta) in header.as_object().ok_or("header not an object")? {
            if name == "__metadata__" {
                continue;
            }
            let dtype = meta["dtype"].as_str().ok_or("tensor dtype missing")?;
            let shape: Vec<usize> = meta["shape"]
                .as_array()
                .ok_or("tensor shape missing")?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = meta["data_offsets"]
                .as_array()
                .ok_or("data_offsets missing")?;
            let (a, b) = (
                off[0].as_u64().ok_or("offset")? as usize,
                off[1].as_u64().ok_or("offset")? as usize,
            );
            let raw = data
                .get(a..b)
                .ok_or_else(|| format!("{name}: offsets outside file"))?;
            let vals: Vec<f32> = match dtype {
                "F32" => raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                "BF16" => raw
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect(),
                "F16" => raw
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect(),
                other => return Err(format!("{name}: unsupported dtype {other}")),
            };
            if vals.len() != shape.iter().product::<usize>() {
                return Err(format!("{name}: {} values for shape {shape:?}", vals.len()));
            }
            map.insert(name.clone(), (shape, vals));
        }
        Ok(Self { map })
    }

    fn merge(&mut self, other: Tensors) {
        for (k, v) in other.map {
            self.map.entry(k).or_insert(v);
        }
    }

    fn get(&self, name: &str, expect: usize) -> Result<&[f32], String> {
        let (_, v) = self
            .map
            .get(name)
            .ok_or_else(|| format!("fixture tensor `{name}` missing"))?;
        if v.len() != expect {
            return Err(format!(
                "fixture tensor `{name}` has {} values, expected {expect}",
                v.len()
            ));
        }
        Ok(v)
    }
}

// ── metrics ─────────────────────────────────────────────────────────────────

struct Cmp {
    ok: bool,
    json: Value,
}

/// Acceptance band of a `compare`: relative L2, cosine, and the max-abs gate as a
/// fraction of max|reference| (catches localized corruption).
#[derive(Clone, Copy)]
struct Band {
    rel_l2: f64,
    cosine: f64,
    max_abs_frac: f64,
}

const BF16_BAND: Band = Band {
    rel_l2: BF16_REL_L2,
    cosine: BF16_COSINE,
    max_abs_frac: BF16_MAX_ABS_FRAC,
};

/// Q8 mirror vs the unquantized truth (fixture / CPU reference): the mirror's own
/// quantization noise (INT8 codes, one F16 scale per 128 channels: ~0.4% rms per
/// element, ~1% rms of the attention output on typical rows, ~2.3% on the
/// extreme-range `*_spread` cases) on top of the BF16 band's arithmetic noise.
const Q8_TRUTH_BAND: Band = Band {
    rel_l2: 3e-2,
    cosine: 0.999,
    max_abs_frac: 0.10,
};

/// Tiled attention on a Q8 mirror vs the reference route on the same dequantized
/// values (the decode kernel keeps `code * scale` in F32 where the reference
/// reads it BF16-rounded, which the `*_spread` cases' huge logits amplify).
const Q8_REF_BAND: Band = Band {
    rel_l2: BF16_REL_L2,
    cosine: BF16_COSINE,
    max_abs_frac: 5e-2,
};

fn compare(name: &str, got: &[f32], want: &[f32]) -> Cmp {
    compare_in(name, got, want, BF16_BAND)
}

fn compare_in(name: &str, got: &[f32], want: &[f32], band: Band) -> Cmp {
    let (mut num, mut den, mut dot, mut gg, mut max_abs, mut max_ref) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    let mut nonfinite = 0usize;
    for (g, w) in got.iter().zip(want) {
        if !g.is_finite() {
            nonfinite += 1;
        }
        let (g, w) = (*g as f64, *w as f64);
        let d = g - w;
        num += d * d;
        den += w * w;
        dot += g * w;
        gg += g * g;
        max_abs = max_abs.max(d.abs());
        max_ref = max_ref.max(w.abs());
    }
    let rel = if den > 0.0 {
        (num / den).sqrt()
    } else {
        num.sqrt()
    };
    let cos = if den > 0.0 && gg > 0.0 {
        dot / (den.sqrt() * gg.sqrt())
    } else {
        1.0
    };
    let tol = 2e-5 + band.max_abs_frac * max_ref;
    let first = got
        .iter()
        .zip(want)
        .position(|(g, w)| !g.is_finite() || ((*g - *w).abs() as f64) > tol);
    let ok = got.len() == want.len()
        && nonfinite == 0
        && rel <= band.rel_l2
        && cos >= band.cosine
        && first.is_none();
    let mut j = json!({
        "tensor": name, "pass": ok, "rel_l2": rel, "cosine": cos, "max_abs": max_abs,
        "max_abs_ref": max_ref, "band": {"rel_l2": band.rel_l2, "cosine": band.cosine,
        "max_abs_tol": tol}, "len": got.len(), "nonfinite": nonfinite,
    });
    if let Some(i) = first {
        j["first_mismatch"] = json!({"index": i, "got": got[i], "want": want[i]});
    }
    Cmp { ok, json: j }
}

fn exact(name: &str, a: &[u8], b: &[u8]) -> Cmp {
    let first = a.iter().zip(b).position(|(x, y)| x != y);
    let ok = a.len() == b.len() && first.is_none();
    let mut j = json!({"tensor": name, "pass": ok, "bitwise": true, "bytes": a.len()});
    if let Some(i) = first {
        j["first_mismatch"] = json!({"byte": i, "a": a[i], "b": b[i]});
    } else if a.len() != b.len() {
        j["first_mismatch"] = json!({"len_a": a.len(), "len_b": b.len()});
    }
    Cmp { ok, json: j }
}

// ── device helpers ──────────────────────────────────────────────────────────

fn he<T: std::fmt::Display>(e: T) -> String {
    e.to_string()
}

fn up_f32(gpu: &mut Gpu, v: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.upload_f32(v, shape).map_err(he)
}

fn up_bf16(gpu: &mut Gpu, v: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    let t = gpu.alloc_tensor(shape, DType::BF16).map_err(he)?;
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|f| f32_to_bf16(*f).to_le_bytes())
        .collect();
    gpu.memcpy_htod_auto(&t.buf, &bytes).map_err(he)?;
    Ok(t)
}

fn poison_bf16(gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
    let t = gpu.alloc_tensor(shape, DType::BF16).map_err(he)?;
    let n: usize = shape.iter().product();
    let bytes: Vec<u8> = (0..n).flat_map(|_| POISON_BF16.to_le_bytes()).collect();
    gpu.memcpy_htod_auto(&t.buf, &bytes).map_err(he)?;
    Ok(t)
}

fn bytes_bf16(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn set_pos(gpu: &mut Gpu, pos: &GpuTensor, p: usize) -> Result<(), String> {
    gpu.memcpy_htod_auto(&pos.buf, &(p as i32).to_le_bytes())
        .map_err(he)
}

/// Token-major mirrors `knope[t][h][nope]`, `vmir[t][h][v]` → `[rows, H, nope+v]`
/// F32 in the reference layout.
fn expand_mirror(k: &[u8], v: &[u8], g: &LatentGeometry, rows: usize) -> Vec<f32> {
    let kf = bytes_bf16(k);
    let vf = bytes_bf16(v);
    let per = g.nope + g.v_dim;
    let mut out = vec![0f32; rows * g.n_heads * per];
    for t in 0..rows {
        for h in 0..g.n_heads {
            let dst = (t * g.n_heads + h) * per;
            let ks = (t * g.n_heads + h) * g.nope;
            out[dst..dst + g.nope].copy_from_slice(&kf[ks..ks + g.nope]);
            let vs = (t * g.n_heads + h) * g.v_dim;
            out[dst + g.nope..dst + per].copy_from_slice(&vf[vs..vs + g.v_dim]);
        }
    }
    out
}

/// Bytes of the valid rows of one token-major mirror (`[cap][H][w]`, rows `0..rows`).
fn valid_rows(bytes: &[u8], heads: usize, w: usize, rows: usize) -> Vec<u8> {
    bytes[..rows * heads * w * 2].to_vec()
}

/// Label a check with the attention route it exercises and whether it is a
/// bit-exact (byte-equality) claim.
fn labeled(mut c: Cmp, route: &str, bit_exact: bool) -> Cmp {
    c.json["route"] = json!(route);
    c.json["bit_exact"] = json!(bit_exact);
    c
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn bf16r(f: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(f))
}

// ── Q8 mirror host model ────────────────────────────────────────────────────

/// Byte the Q8 planes are poisoned with (any non-zero pattern).
const Q8_POISON: u8 = 0x77;

/// IEEE binary32 → binary16, round-to-nearest-even (subnormals, overflow → inf).
fn f32_to_f16_rne(f: f32) -> u16 {
    let u = f.to_bits();
    let sign = ((u >> 16) & 0x8000) as u16;
    let exp = ((u >> 23) & 0xff) as i32;
    let man = u & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let tie = 1u32 << (shift - 1);
        let up = rem > tie || (rem == tie && (half & 1) == 1);
        return sign | (half + up as u32) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let up = rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1);
    sign | (half + up as u32) as u16
}

/// The production Q8 quantizer for one (row, head, plane) vector: F16 RNE
/// scale `min(max|x|/127, 65504)`, codes `clamp(rint(x / scale), ±127)`.
fn q8_quant_vec(x: &[f32]) -> (Vec<i8>, u16) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let s16 = f32_to_f16_rne((amax / 127.0f32).min(65504.0));
    let sf = f16_to_f32(s16);
    let codes = x
        .iter()
        .map(|v| {
            if sf > 0.0 {
                (v / sf).round_ties_even().clamp(-127.0, 127.0) as i8
            } else {
                0
            }
        })
        .collect();
    (codes, s16)
}

/// Host Q8 plane of `rows` rows: `bytes` is the production layout (per row
/// `heads*width` int8 codes then `heads` LE F16 scales); `deq` is the BF16 (LE
/// bytes, `[rows][heads][width]`) dequantization `bf16r(code * scale)` the
/// prefill kernel feeds to the matrix units.
struct Q8Plane {
    bytes: Vec<u8>,
    deq: Vec<u8>,
}

/// `src(vec_index, out)` fills `out` (`width` f32) with the vector of
/// (row, head) = (`vec_index / heads`, `vec_index % heads`).
fn q8_plane(
    rows: usize,
    heads: usize,
    width: usize,
    src: &(dyn Fn(usize, &mut [f32]) + Sync),
) -> Q8Plane {
    let rb = q8_plane_row_bytes(heads, width);
    let db = heads * width * 2;
    let mut bytes = vec![0u8; rows * rb];
    let mut deq = vec![0u8; rows * db];
    if rows == 0 {
        return Q8Plane { bytes, deq };
    }
    let per = rows.div_ceil(worker_count().min(rows));
    std::thread::scope(|sc| {
        for (ci, (bc, dc)) in bytes
            .chunks_mut(per * rb)
            .zip(deq.chunks_mut(per * db))
            .enumerate()
        {
            sc.spawn(move || {
                let mut scratch = vec![0f32; width];
                for r in 0..bc.len() / rb {
                    let row = ci * per + r;
                    for hh in 0..heads {
                        src(row * heads + hh, &mut scratch);
                        let (codes, s16) = q8_quant_vec(&scratch);
                        let sf = f16_to_f32(s16);
                        let brow = &mut bc[r * rb..(r + 1) * rb];
                        let drow = &mut dc[r * db..(r + 1) * db];
                        for (e, c) in codes.iter().enumerate() {
                            brow[hh * width + e] = *c as u8;
                            let d = f32_to_bf16(*c as f32 * sf).to_le_bytes();
                            drow[(hh * width + e) * 2..(hh * width + e) * 2 + 2]
                                .copy_from_slice(&d);
                        }
                        let so = heads * width + 2 * hh;
                        brow[so..so + 2].copy_from_slice(&s16.to_le_bytes());
                    }
                }
            });
        }
    });
    Q8Plane { bytes, deq }
}

/// Q8 plane of a token-major BF16 mirror (`[rows][heads][width]`, LE bytes).
fn q8_plane_from_bf16(b: &[u8], rows: usize, heads: usize, width: usize) -> Q8Plane {
    q8_plane(rows, heads, width, &|vi, out| {
        let off = vi * width * 2;
        for (e, o) in out.iter_mut().enumerate() {
            *o = bf16_to_f32(u16::from_le_bytes([b[off + 2 * e], b[off + 2 * e + 1]]));
        }
    })
}

/// `[cap_rows, row_bytes]` Raw plane filled with [`Q8_POISON`].
fn poison_raw(gpu: &mut Gpu, cap_rows: usize, row_bytes: usize) -> Result<GpuTensor, String> {
    upload_raw_padded(gpu, &[], cap_rows, row_bytes)
}

/// `[cap_rows, row_bytes]` Raw plane holding `prefix` then [`Q8_POISON`].
fn upload_raw_padded(
    gpu: &mut Gpu,
    prefix: &[u8],
    cap_rows: usize,
    row_bytes: usize,
) -> Result<GpuTensor, String> {
    let t = gpu
        .alloc_tensor(&[cap_rows, row_bytes], DType::Raw)
        .map_err(he)?;
    let mut buf = Vec::with_capacity(cap_rows * row_bytes);
    buf.extend_from_slice(prefix);
    buf.resize(cap_rows * row_bytes, Q8_POISON);
    gpu.memcpy_htod_auto(&t.buf, &buf).map_err(he)?;
    Ok(t)
}

/// Plain BF16 mirror plane `[cap_rows, heads, width]` holding `prefix` (BF16 LE
/// bytes) then poison.
fn upload_bf16_padded(gpu: &mut Gpu, prefix: &[u8], shape: &[usize]) -> Result<GpuTensor, String> {
    let t = poison_bf16(gpu, shape)?;
    gpu.memcpy_htod_auto(&t.buf, prefix).map_err(he)?;
    Ok(t)
}

/// Tiled attention outputs (twice) on a Q8 mirror and the Reference output on
/// its dequantization.
struct Q8Attn {
    y1: Vec<f32>,
    y2: Vec<f32>,
    yref: Vec<f32>,
}

/// Quantize the dense BF16 mirror (`knope`/`vmir`, valid rows `0..total`) on
/// host, upload it as Q8 planes (capacity `mcap`) plus its dequantized BF16
/// mirror, and run Tiled (twice) on the former and Reference on the latter.
/// Frees everything it allocates, also on error.
#[allow(clippy::too_many_arguments)]
fn q8_mirror_attention(
    gpu: &mut Gpu,
    g: &LatentGeometry,
    cache: LatentCache<'_>,
    rd: &RouteDev,
    knope: &GpuTensor,
    vmir: &GpuTensor,
    total: usize,
    mcap: usize,
    d_q: &GpuTensor,
    n: usize,
    scale: f32,
) -> Result<Q8Attn, String> {
    let (h, nope, v) = (g.n_heads, g.nope, g.v_dim);
    let kb = gpu.download_raw_bytes(knope).map_err(he)?;
    let vb = gpu.download_raw_bytes(vmir).map_err(he)?;
    let kq = q8_plane_from_bf16(&kb, total, h, nope);
    let vq = q8_plane_from_bf16(&vb, total, h, v);
    drop((kb, vb));
    let (rbk, rbv) = (q8_plane_row_bytes(h, nope), q8_plane_row_bytes(h, v));
    let d_kq = upload_raw_padded(gpu, &kq.bytes, mcap, rbk)?;
    let d_vq = upload_raw_padded(gpu, &vq.bytes, mcap, rbv)?;
    let d_kd = upload_bf16_padded(gpu, &kq.deq, &[mcap, h, nope])?;
    let d_vd = upload_bf16_padded(gpu, &vq.deq, &[mcap, h, v])?;
    drop((kq, vq));
    let d_y1 = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let d_y2 = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let d_yr = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let outcome = (|| -> Result<Q8Attn, String> {
        let m8 = ExpandedMirror {
            knope: &d_kq,
            vmir: &d_vq,
            capacity_rows: mcap,
            q8: true,
        };
        let md = ExpandedMirror {
            knope: &d_kd,
            vmir: &d_vd,
            capacity_rows: mcap,
            q8: false,
        };
        for y in [&d_y1, &d_y2] {
            attn(gpu, g, cache, Some(m8), d_q, y, n, scale, rd.tiled())?;
        }
        attn(
            gpu,
            g,
            cache,
            Some(md),
            d_q,
            &d_yr,
            n,
            scale,
            AttentionRoute::Reference,
        )?;
        gpu.hip.device_synchronize().map_err(he)?;
        Ok(Q8Attn {
            y1: gpu.download_f32(&d_y1).map_err(he)?,
            y2: gpu.download_f32(&d_y2).map_err(he)?,
            yref: gpu.download_f32(&d_yr).map_err(he)?,
        })
    })();
    for t in [d_kq, d_vq, d_kd, d_vd, d_y1, d_y2, d_yr] {
        let _ = gpu.free_tensor(t);
    }
    outcome
}

/// Which non-reference routes run (the reference route always runs first).
#[derive(Clone, Copy)]
struct Routes {
    tiled: bool,
    absorbed: bool,
}

impl Routes {
    fn parse(opts: &BTreeMap<String, String>) -> Result<Self, String> {
        match opts.get("route").map(String::as_str).unwrap_or("all") {
            "all" => Ok(Self {
                tiled: true,
                absorbed: true,
            }),
            "reference" => Ok(Self {
                tiled: false,
                absorbed: false,
            }),
            "tiled" => Ok(Self {
                tiled: true,
                absorbed: false,
            }),
            "absorbed" => Ok(Self {
                tiled: false,
                absorbed: true,
            }),
            other => Err(format!(
                "--route `{other}`: expected reference|tiled|absorbed|all"
            )),
        }
    }

    fn names(self) -> Vec<&'static str> {
        let mut v = vec!["reference"];
        if self.tiled {
            v.push("tiled");
            v.push("tiled_q8");
        }
        if self.absorbed {
            v.push("absorbed");
        }
        v
    }
}

/// Device operands of the non-reference routes: the split-KV decode workspace
/// and the absorbed operands (packed expansion weight + q_abs/z planes).
struct RouteDev {
    ws: Option<GpuTensor>,
    packed: Option<GpuTensor>,
    q_abs: Option<GpuTensor>,
    z: Option<GpuTensor>,
}

impl RouteDev {
    /// `kv_b`: `[H*(nope+v), latent]` expansion weight (BF16-rounded here, as
    /// the device decompression reads it).
    fn new(
        gpu: &mut Gpu,
        g: &LatentGeometry,
        rows: usize,
        routes: Routes,
        kv_b: &[f32],
    ) -> Result<Self, String> {
        let mut me = Self {
            ws: None,
            packed: None,
            q_abs: None,
            z: None,
        };
        if routes.tiled || routes.absorbed {
            let elems = latent_attention_decode_workspace_elems(g.n_heads, g.latent.max(g.v_dim));
            me.ws = Some(gpu.zeros(&[elems], DType::F32).map_err(he)?);
        }
        if routes.absorbed {
            let (h, latent) = (g.n_heads, g.latent);
            let n_out = h * (g.nope + g.v_dim);
            if kv_b.len() != n_out * latent {
                return Err(format!(
                    "absorbed pack: kv_b has {} values, expected {}",
                    kv_b.len(),
                    n_out * latent
                ));
            }
            // w_t row c = column c of the expansion weight W[n_out, latent].
            let mut wt = vec![0f32; latent * n_out];
            for o in 0..n_out {
                for c in 0..latent {
                    wt[c * n_out + o] = bf16r(kv_b[o * latent + c]);
                }
            }
            let d_wt = up_f32(gpu, &wt, &[latent, n_out])?;
            let elems = latent_attention_absorb_packed_elems(h, g.nope, g.v_dim, latent);
            let packed = gpu.zeros(&[elems], DType::BF16).map_err(he)?;
            gpu.latent_attention_absorb_pack(&d_wt, &packed, h, g.nope, g.v_dim, latent)
                .map_err(he)?;
            gpu.hip.device_synchronize().map_err(he)?;
            gpu.free_tensor(d_wt).map_err(he)?;
            me.packed = Some(packed);
            me.q_abs = Some(
                gpu.zeros(&[rows, h, latent + g.tail], DType::F32)
                    .map_err(he)?,
            );
            me.z = Some(gpu.zeros(&[rows, h, latent], DType::F32).map_err(he)?);
        }
        Ok(me)
    }

    fn tiled(&self) -> AttentionRoute<'_> {
        AttentionRoute::Tiled {
            workspace: self.ws.as_ref(),
        }
    }

    fn absorbed(&self) -> Result<AttentionRoute<'_>, String> {
        match (&self.packed, &self.q_abs, &self.z) {
            (Some(packed), Some(q_abs), Some(z)) => Ok(AttentionRoute::Absorbed {
                ops: AbsorbOperands { packed, q_abs, z },
                workspace: self.ws.as_ref(),
            }),
            _ => Err("absorbed route operands were not built".into()),
        }
    }

    fn free(self, gpu: &mut Gpu) {
        for t in [self.ws, self.packed, self.q_abs, self.z]
            .into_iter()
            .flatten()
        {
            let _ = gpu.free_tensor(t);
        }
    }
}

/// One attention stage on the selected route.
#[allow(clippy::too_many_arguments)]
fn attn(
    gpu: &mut Gpu,
    g: &LatentGeometry,
    cache: LatentCache<'_>,
    mirror: Option<ExpandedMirror<'_>>,
    q: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    scale: f32,
    route: AttentionRoute<'_>,
) -> Result<(), String> {
    execute_latent_attention(
        gpu,
        &LatentAttentionOp::Attention {
            geometry: *g,
            cache,
            mirror,
            q,
            y,
            rows,
            scale,
            route,
        },
    )
    .map_err(he)
}

fn dim(dims: &Value, k: &str) -> Result<usize, String> {
    dims.get(k)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("manifest dims missing `{k}`"))
}

// ── one case ────────────────────────────────────────────────────────────────

fn run_case(gpu: &mut Gpu, dir: &Path, case: &Value, routes: Routes) -> Result<Value, String> {
    let name = case["name"]
        .as_str()
        .ok_or("case name missing")?
        .to_string();
    if case.get("mode").and_then(Value::as_str).unwrap_or("bf16") != "bf16" {
        return Err(format!(
            "case {name}: only mode \"bf16\" is implemented for mla"
        ));
    }
    let dims = case.get("dims").ok_or("case dims missing")?;
    let (n, p) = (dim(dims, "N")?, dim(dims, "P")?);
    let g = LatentGeometry {
        n_heads: dim(dims, "H")?,
        nope: dim(dims, "nope")?,
        tail: dim(dims, "tail")?,
        v_dim: dim(dims, "v")?,
        latent: dim(dims, "latent")?,
    };
    let qlora = dim(dims, "qlora")?;
    let file = case["file"].as_str().ok_or("case file missing")?;
    let mut t = Tensors::load(&dir.join(file))?;
    if let Some(w) = case.get("weights").and_then(Value::as_str) {
        t.merge(Tensors::load(&dir.join(w))?);
    }
    let (width, dqk) = (g.row_width(), g.qk_dim());
    let (h, v, per) = (g.n_heads, g.v_dim, g.nope + g.v_dim);
    let total = p + n;
    let cap = total + 37; // latent-cache rows (poison beyond the live length)
    let mcap = total + 11; // mirror rows: independent of the cache capacity
    let scale = g.default_scale();

    let q_a_out = t.get("q_a_out", n * qlora)?.to_vec();
    let kv_a_out = t.get("kv_a_out", n * width)?.to_vec();
    let q_a_norm = t.get("q_a_norm", qlora)?.to_vec();
    let kv_a_norm = t.get("kv_a_norm", g.latent)?.to_vec();
    let q_proj = t.get("q_proj", n * h * dqk)?.to_vec();
    let kv_b = t.get("kv_b", h * per * g.latent)?.to_vec();
    let g_out = t.get("g_out", n * h * v)?.to_vec();
    let cache_prefix = if p == 0 {
        Vec::new()
    } else {
        t.get("cache_prefix", p * width)?.to_vec()
    };
    let r_qlat = t.get("qlat", n * qlora)?;
    let r_cache = t.get("cache_out", total * width)?;
    let r_exp = t.get("kv_expanded", total * h * per)?;
    let r_y = t.get("attn_y", n * h * v)?;
    let r_gated = t.get("gated", n * h * v)?;

    // Device state: poisoned cache/mirrors; prefix rows uploaded as BF16.
    let cache_t = poison_bf16(gpu, &[cap, width])?;
    if p > 0 {
        let bytes: Vec<u8> = cache_prefix
            .iter()
            .flat_map(|f| f32_to_bf16(*f).to_le_bytes())
            .collect();
        gpu.memcpy_htod_auto(&cache_t.buf, &bytes).map_err(he)?;
    }
    let knope = poison_bf16(gpu, &[mcap, h, g.nope])?;
    let vmir = poison_bf16(gpu, &[mcap, h, v])?;
    let pos = gpu.alloc_tensor(&[1], DType::F32).map_err(he)?;
    set_pos(gpu, &pos, p)?;

    let d_qa = up_f32(gpu, &q_a_out, &[n, qlora])?;
    let d_kva = up_f32(gpu, &kv_a_out, &[n, width])?;
    let d_qg = up_f32(gpu, &q_a_norm, &[qlora])?;
    let d_kg = up_bf16(gpu, &kv_a_norm, &[g.latent])?; // BF16 gamma path
    let d_q = up_f32(gpu, &q_proj, &[n, h * dqk])?;
    let d_w = up_bf16(gpu, &kv_b, &[h * per, g.latent])?;
    let d_g = up_f32(gpu, &g_out, &[n, h * v])?;
    let d_qlat = gpu.zeros(&[n, qlora], DType::F32).map_err(he)?;
    let d_y = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let d_y2 = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let d_gated = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let d_yt1 = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let d_yt2 = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let d_gated_t = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let d_ya = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;

    let cache = LatentCache {
        rows: &cache_t,
        capacity_rows: cap,
        pos: &pos,
        host_pos: Some(p),
    };
    let mirror = ExpandedMirror {
        knope: &knope,
        vmir: &vmir,
        capacity_rows: mcap,
        q8: false,
    };
    let dense = KvExpansion::Dense { weight: &d_w };
    let run = |gpu: &mut Gpu, op: LatentAttentionOp<'_>| -> Result<(), String> {
        execute_latent_attention(gpu, &op).map_err(he)
    };

    // 1. q latent norm; 2. split into cache at device position P.
    run(
        gpu,
        LatentAttentionOp::NormRows {
            x: &d_qa,
            gamma: &d_qg,
            out: &d_qlat,
            rows: n,
            width: qlora,
            eps: 1e-6,
        },
    )?;
    run(
        gpu,
        LatentAttentionOp::SplitKv {
            geometry: g,
            cache,
            kv_a: &d_kva,
            gamma: &d_kg,
            rows: n,
            eps: 1e-6,
        },
    )?;
    // 3. schedule A: prefix rows in 37-row absolute tiles, new rows via device base.
    let mut s = 0;
    while s < p {
        let c = 37.min(p - s);
        run(
            gpu,
            LatentAttentionOp::Decompress {
                geometry: g,
                cache,
                mirror,
                expansion: dense,
                range: RowRange::Absolute { start: s, count: c },
            },
        )?;
        s += c;
    }
    run(
        gpu,
        LatentAttentionOp::Decompress {
            geometry: g,
            cache,
            mirror,
            expansion: dense,
            range: RowRange::NewRows { count: n },
        },
    )?;
    // 4. attention (twice, for byte determinism) and gate.
    for y in [&d_y, &d_y2] {
        attn(
            gpu,
            &g,
            cache,
            Some(mirror),
            &d_q,
            y,
            n,
            scale,
            AttentionRoute::Reference,
        )?;
    }
    run(
        gpu,
        LatentAttentionOp::Gate {
            y: &d_y,
            g: &d_g,
            out: &d_gated,
            elements: n * h * v,
        },
    )?;
    // 4b. tiled (same mirror) and absorbed (no mirror) routes.
    let rd = RouteDev::new(gpu, &g, n, routes, &kv_b)?;
    if routes.tiled {
        for y in [&d_yt1, &d_yt2] {
            attn(gpu, &g, cache, Some(mirror), &d_q, y, n, scale, rd.tiled())?;
        }
        run(
            gpu,
            LatentAttentionOp::Gate {
                y: &d_yt1,
                g: &d_g,
                out: &d_gated_t,
                elements: n * h * v,
            },
        )?;
    }
    if routes.absorbed {
        attn(gpu, &g, cache, None, &d_q, &d_ya, n, scale, rd.absorbed()?)?;
    }

    gpu.hip.device_synchronize().map_err(he)?;
    let qlat = gpu.download_f32(&d_qlat).map_err(he)?;
    let cache_bytes = gpu.download_raw_bytes(&cache_t).map_err(he)?;
    let k_a = gpu.download_raw_bytes(&knope).map_err(he)?;
    let v_a = gpu.download_raw_bytes(&vmir).map_err(he)?;
    let y1 = gpu.download_f32(&d_y).map_err(he)?;
    let y2 = gpu.download_f32(&d_y2).map_err(he)?;
    let gated = gpu.download_f32(&d_gated).map_err(he)?;
    let (yt1, yt2, gated_t) = if routes.tiled {
        (
            gpu.download_f32(&d_yt1).map_err(he)?,
            gpu.download_f32(&d_yt2).map_err(he)?,
            gpu.download_f32(&d_gated_t).map_err(he)?,
        )
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
    let ya = if routes.absorbed {
        gpu.download_f32(&d_ya).map_err(he)?
    } else {
        Vec::new()
    };

    let cache_f = bytes_bf16(&cache_bytes);
    let mut checks: Vec<Cmp> = Vec::new();
    checks.push(compare("qlat", &qlat, r_qlat));
    checks.push(compare("cache_out", &cache_f[..total * width], r_cache));
    // Prefix rows must be untouched (bit-preserved) by the split.
    let prefix_expect: Vec<u8> = cache_prefix
        .iter()
        .flat_map(|f| f32_to_bf16(*f).to_le_bytes())
        .collect();
    checks.push(exact(
        "cache_prefix_preserved",
        &cache_bytes[..p * width * 2],
        &prefix_expect,
    ));
    // Tail channels of the new rows are the raw kv_a tail (never normalized).
    let new_tail: Vec<u8> = (0..n)
        .flat_map(|i| {
            kv_a_out[i * width + g.latent..(i + 1) * width]
                .iter()
                .flat_map(|f| f32_to_bf16(*f).to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect();
    let got_tail: Vec<u8> = (0..n)
        .flat_map(|i| {
            let r = p + i;
            cache_bytes[(r * width + g.latent) * 2..((r + 1) * width) * 2].to_vec()
        })
        .collect();
    checks.push(exact("cache_tail_raw", &got_tail, &new_tail));
    let exp_a = expand_mirror(&k_a, &v_a, &g, total);
    checks.push(compare("kv_expanded", &exp_a, r_exp));
    checks.push(labeled(compare("attn_y", &y1, r_y), "reference", false));
    checks.push(labeled(
        compare("gated", &gated, r_gated),
        "reference",
        false,
    ));
    checks.push(labeled(
        exact("attn_y_repeat_bytewise", &f32_bytes(&y1), &f32_bytes(&y2)),
        "reference",
        true,
    ));
    if routes.tiled {
        checks.push(labeled(compare("attn_y_tiled", &yt1, r_y), "tiled", false));
        checks.push(labeled(
            exact(
                "attn_y_tiled_repeat_bytewise",
                &f32_bytes(&yt1),
                &f32_bytes(&yt2),
            ),
            "tiled",
            true,
        ));
        checks.push(labeled(
            compare("gated_tiled", &gated_t, r_gated),
            "tiled",
            false,
        ));
    }
    if routes.absorbed {
        // The absorbed route computes the unrounded math (no intermediate BF16
        // K/V), so it is gated against the unrounded-f64 CPU reference; the
        // fixture's own BF16-K/V numbers are reported, non-gating.
        let wq: Vec<f32> = kv_b.iter().map(|f| bf16r(*f)).collect();
        let data = LongData {
            cache: r_cache.to_vec(),
            q: q_proj.clone(),
            planted: Vec::new(),
        };
        let exact_y = cpu_reference(&g, p, n, &data, &wq, true).y;
        checks.push(labeled(
            compare("attn_y_absorbed", &ya, &exact_y),
            "absorbed",
            false,
        ));
        checks.push(informational(labeled(
            compare("attn_y_absorbed_vs_fixture", &ya, r_y),
            "absorbed",
            false,
        )));
    }
    if routes.tiled && routes.absorbed {
        checks.push(labeled(
            band_no_outlier(compare("absorbed_vs_tiled", &ya, &yt1)),
            "absorbed",
            false,
        ));
    }

    // 5. schedule B: every live row in one absolute call, fresh mirrors.
    let kn_b = poison_bf16(gpu, &[mcap, h, g.nope])?;
    let vm_b = poison_bf16(gpu, &[mcap, h, v])?;
    let mirror_b = ExpandedMirror {
        knope: &kn_b,
        vmir: &vm_b,
        capacity_rows: mcap,
        q8: false,
    };
    run(
        gpu,
        LatentAttentionOp::Decompress {
            geometry: g,
            cache,
            mirror: mirror_b,
            expansion: dense,
            range: RowRange::Absolute {
                start: 0,
                count: total,
            },
        },
    )?;
    gpu.hip.device_synchronize().map_err(he)?;
    let k_b = gpu.download_raw_bytes(&kn_b).map_err(he)?;
    let v_b = gpu.download_raw_bytes(&vm_b).map_err(he)?;
    checks.push(exact(
        "mirror_k_schedules_bitwise",
        &valid_rows(&k_a, h, g.nope, total),
        &valid_rows(&k_b, h, g.nope, total),
    ));
    checks.push(exact(
        "mirror_v_schedules_bitwise",
        &valid_rows(&v_a, h, v, total),
        &valid_rows(&v_b, h, v, total),
    ));

    // 5b. Q8 mirror attention built from the dense mirror (schedule A): tiled
    // on the Q8 planes vs Reference on their dequantization vs the fixture.
    if routes.tiled {
        gpu.hip.device_synchronize().map_err(he)?;
        let q8 = q8_mirror_attention(
            gpu, &g, cache, &rd, &knope, &vmir, total, mcap, &d_q, n, scale,
        )?;
        checks.push(labeled(
            exact(
                "attn_y_tiled_q8_repeat_bytewise",
                &f32_bytes(&q8.y1),
                &f32_bytes(&q8.y2),
            ),
            "tiled_q8",
            true,
        ));
        checks.push(labeled(
            compare_in(
                "attn_y_tiled_q8_vs_q8_reference",
                &q8.y1,
                &q8.yref,
                Q8_REF_BAND,
            ),
            "tiled_q8",
            false,
        ));
        checks.push(labeled(
            compare_in("attn_y_tiled_q8_vs_fixture", &q8.y1, r_y, Q8_TRUTH_BAND),
            "tiled_q8",
            false,
        ));
    }

    // 6. gather -> external projector (host, ascending-k fma) -> scatter.
    if total <= EXTERNAL_ROW_LIMIT {
        let plane = gpu.zeros(&[total, g.latent], DType::F32).map_err(he)?;
        run(
            gpu,
            LatentAttentionOp::GatherLatent {
                geometry: g,
                cache,
                range: RowRange::Absolute {
                    start: 0,
                    count: total,
                },
                out: &plane,
            },
        )?;
        gpu.hip.device_synchronize().map_err(he)?;
        let lat = gpu.download_f32(&plane).map_err(he)?;
        let wq: Vec<f32> = kv_b.iter().map(|f| bf16_to_f32(f32_to_bf16(*f))).collect();
        let mut proj = vec![0f32; total * h * per];
        for r in 0..total {
            for o in 0..h * per {
                let mut acc = 0f32;
                for k in 0..g.latent {
                    acc = lat[r * g.latent + k].mul_add(wq[o * g.latent + k], acc);
                }
                proj[r * h * per + o] = acc;
            }
        }
        let d_proj = up_f32(gpu, &proj, &[total, h * per])?;
        let kn_c = poison_bf16(gpu, &[mcap, h, g.nope])?;
        let vm_c = poison_bf16(gpu, &[mcap, h, v])?;
        let mirror_c = ExpandedMirror {
            knope: &kn_c,
            vmir: &vm_c,
            capacity_rows: mcap,
            q8: false,
        };
        run(
            gpu,
            LatentAttentionOp::Decompress {
                geometry: g,
                cache,
                mirror: mirror_c,
                expansion: KvExpansion::External { projected: &d_proj },
                range: RowRange::Absolute {
                    start: 0,
                    count: total,
                },
            },
        )?;
        gpu.hip.device_synchronize().map_err(he)?;
        let k_c = gpu.download_raw_bytes(&kn_c).map_err(he)?;
        let v_c = gpu.download_raw_bytes(&vm_c).map_err(he)?;
        checks.push(exact(
            "mirror_k_external_bitwise",
            &valid_rows(&k_a, h, g.nope, total),
            &valid_rows(&k_c, h, g.nope, total),
        ));
        checks.push(exact(
            "mirror_v_external_bitwise",
            &valid_rows(&v_a, h, v, total),
            &valid_rows(&v_c, h, v, total),
        ));

        // 6b. Q8 mirror scatter of the same projector output: bytes must equal
        // the host quantizer row by row, under two schedules, poison untouched.
        if routes.tiled {
            let (rbk, rbv) = (q8_plane_row_bytes(h, g.nope), q8_plane_row_bytes(h, v));
            let proj_k = |vi: usize, out: &mut [f32]| {
                let o = (vi / h) * h * per + (vi % h) * per;
                out.copy_from_slice(&proj[o..o + g.nope]);
            };
            let proj_v = |vi: usize, out: &mut [f32]| {
                let o = (vi / h) * h * per + (vi % h) * per + g.nope;
                out.copy_from_slice(&proj[o..o + v]);
            };
            let want_k = q8_plane(total, h, g.nope, &proj_k).bytes;
            let want_v = q8_plane(total, h, v, &proj_v).bytes;
            let kq1 = poison_raw(gpu, mcap, rbk)?;
            let vq1 = poison_raw(gpu, mcap, rbv)?;
            let kq2 = poison_raw(gpu, mcap, rbk)?;
            let vq2 = poison_raw(gpu, mcap, rbv)?;
            let m1 = ExpandedMirror {
                knope: &kq1,
                vmir: &vq1,
                capacity_rows: mcap,
                q8: true,
            };
            let m2 = ExpandedMirror {
                knope: &kq2,
                vmir: &vq2,
                capacity_rows: mcap,
                q8: true,
            };
            run(
                gpu,
                LatentAttentionOp::Decompress {
                    geometry: g,
                    cache,
                    mirror: m1,
                    expansion: KvExpansion::External { projected: &d_proj },
                    range: RowRange::Absolute {
                        start: 0,
                        count: total,
                    },
                },
            )?;
            // Schedule 2: 37-row absolute tiles over the prefix, new rows via
            // the device base position; each tile's projected plane holds
            // exactly its rows.
            let mut tiles: Vec<GpuTensor> = Vec::new();
            let mut s = 0;
            while s < p {
                let c = 37.min(p - s);
                let tile = up_f32(gpu, &proj[s * h * per..(s + c) * h * per], &[c, h * per])?;
                run(
                    gpu,
                    LatentAttentionOp::Decompress {
                        geometry: g,
                        cache,
                        mirror: m2,
                        expansion: KvExpansion::External { projected: &tile },
                        range: RowRange::Absolute { start: s, count: c },
                    },
                )?;
                tiles.push(tile);
                s += c;
            }
            let tail = up_f32(gpu, &proj[p * h * per..total * h * per], &[n, h * per])?;
            run(
                gpu,
                LatentAttentionOp::Decompress {
                    geometry: g,
                    cache,
                    mirror: m2,
                    expansion: KvExpansion::External { projected: &tail },
                    range: RowRange::NewRows { count: n },
                },
            )?;
            tiles.push(tail);
            gpu.hip.device_synchronize().map_err(he)?;
            let k1 = gpu.download_raw_bytes(&kq1).map_err(he)?;
            let v1 = gpu.download_raw_bytes(&vq1).map_err(he)?;
            let k2 = gpu.download_raw_bytes(&kq2).map_err(he)?;
            let v2 = gpu.download_raw_bytes(&vq2).map_err(he)?;
            checks.push(exact("mirror_q8_k_bitwise", &k1[..total * rbk], &want_k));
            checks.push(exact("mirror_q8_v_bitwise", &v1[..total * rbv], &want_v));
            checks.push(exact("mirror_q8_schedules_bitwise", &k1, &k2));
            checks.push(exact("mirror_q8_schedules_v_bitwise", &v1, &v2));
            let pk = vec![Q8_POISON; (mcap - total) * rbk];
            let pv = vec![Q8_POISON; (mcap - total) * rbv];
            checks.push(exact(
                "mirror_q8_k_poison_untouched",
                &k1[total * rbk..],
                &pk,
            ));
            checks.push(exact(
                "mirror_q8_v_poison_untouched",
                &v1[total * rbv..],
                &pv,
            ));
            for t in tiles.into_iter().chain([kq1, vq1, kq2, vq2]) {
                let _ = gpu.free_tensor(t);
            }
        }
        for t in [plane, d_proj, kn_c, vm_c] {
            let _ = gpu.free_tensor(t);
        }
    }

    rd.free(gpu);
    for t in [
        cache_t, knope, vmir, pos, d_qa, d_kva, d_qg, d_kg, d_q, d_w, d_g, d_qlat, d_y, d_y2,
        d_gated, d_yt1, d_yt2, d_gated_t, d_ya, kn_b, vm_b,
    ] {
        let _ = gpu.free_tensor(t);
    }
    let (pass, check_json) = finish_checks(checks);
    Ok(json!({
        "name": name, "pass": pass,
        "dims": {"N": n, "P": p, "H": h, "nope": g.nope, "tail": g.tail, "v": v,
                 "latent": g.latent, "qlora": qlora},
        "checks": check_json,
    }))
}

// ── generated long-context cases ────────────────────────────────────────────

/// xorshift64* with a cached Box-Muller spare; fully deterministic from the seed.
struct Rng {
    s: u64,
    spare: Option<f64>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            s: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `(0, 1]`.
    fn uniform(&mut self) -> f64 {
        (((self.next_u64() >> 11) + 1) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Standard normal (Box-Muller; the sine branch is cached for the next call).
    fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * th.sin());
        r * th.cos()
    }
}

fn long_seed(p: usize, n: usize) -> u64 {
    LONG_SEED ^ ((p as u64) << 20) ^ (n as u64)
}

/// Generated inputs of one long case (all BF16-valued F32).
struct LongData {
    /// `[P+N, latent+tail]` latent cache rows (needle rows already planted).
    cache: Vec<f32>,
    /// `[N, H, nope+tail]` queries.
    q: Vec<f32>,
    /// Cache rows whose tail carries the needle.
    planted: Vec<usize>,
}

fn gen_long(g: &LatentGeometry, p: usize, n: usize) -> LongData {
    let mut rng = Rng::new(long_seed(p, n));
    let (h, width, dqk) = (g.n_heads, g.row_width(), g.qk_dim());
    let mut cache = vec![0f32; (p + n) * width];
    for x in cache.iter_mut() {
        *x = bf16r(rng.normal() as f32);
    }
    let mut q = vec![0f32; n * h * dqk];
    for x in q.iter_mut() {
        *x = bf16r((LONG_QUERY_STD * rng.normal()) as f32);
    }
    // Needle: tail = gain * mean over heads of the tail of the last query row, so
    // a real softmax peak exists far from the diagonal (rows P/3 and, for N>1, P/2).
    let mut planted = vec![p / 3];
    if n > 1 {
        planted.push(p / 2);
    }
    let needle: Vec<f32> = (0..g.tail)
        .map(|d| {
            let mean = (0..h)
                .map(|hh| q[((n - 1) * h + hh) * dqk + g.nope + d] as f64)
                .sum::<f64>()
                / h as f64;
            bf16r(LONG_NEEDLE_GAIN * mean as f32)
        })
        .collect();
    for &r in &planted {
        cache[r * width + g.latent..(r + 1) * width].copy_from_slice(&needle);
    }
    LongData { cache, q, planted }
}

fn worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(64)
}

/// Report a check but keep it out of the case verdict (`"gating": false`); its
/// own `pass` value is preserved in the JSON.
fn informational(mut c: Cmp) -> Cmp {
    c.json["gating"] = json!(false);
    c.ok = true;
    c
}

/// Gate only on rel_l2 / cosine / finiteness; max_abs is printed but does not
/// gate (`"max_abs_gating": false`).
fn band_no_outlier(mut c: Cmp) -> Cmp {
    let rel = c.json["rel_l2"].as_f64().unwrap_or(f64::INFINITY);
    let cos = c.json["cosine"].as_f64().unwrap_or(0.0);
    let nonfinite = c.json["nonfinite"].as_u64().unwrap_or(1);
    let ok = rel <= BF16_REL_L2 && cos >= BF16_COSINE && nonfinite == 0;
    c.json["pass"] = json!(ok);
    c.json["max_abs_gating"] = json!(false);
    c.ok = ok;
    c
}

/// Case verdict + per-check JSON; every check states whether it gates.
fn finish_checks(checks: Vec<Cmp>) -> (bool, Vec<Value>) {
    let pass = checks.iter().all(|c| c.ok);
    let js = checks
        .into_iter()
        .map(|mut c| {
            if c.json.get("gating").is_none() {
                c.json["gating"] = json!(true);
            }
            c.json
        })
        .collect();
    (pass, js)
}

/// CPU K/V expansion operands. `round_kv`: ascending-index f32 fma accumulation
/// rounded to BF16 like the device decompression; otherwise unrounded (f64
/// accumulation, the mathematical K/V the absorbed route computes).
#[derive(Clone, Copy)]
struct Expand<'a> {
    g: &'a LatentGeometry,
    cache: &'a [f32],
    wq: &'a [f32],
    round_kv: bool,
}

/// K/V rows `r0..r0+rows` of one head.
#[inline(always)]
fn expand_rows_body(x: &Expand<'_>, head: usize, r0: usize, kout: &mut [f32], vout: &mut [f32]) {
    let (g, cache, wq) = (x.g, x.cache, x.wq);
    let (width, latent, per) = (g.row_width(), g.latent, g.nope + g.v_dim);
    let rows = kout.len() / g.nope;
    for rr in 0..rows {
        let lat = &cache[(r0 + rr) * width..(r0 + rr) * width + latent];
        if !x.round_kv {
            for o in 0..per {
                let w = &wq[(head * per + o) * latent..(head * per + o + 1) * latent];
                let mut a = 0f64;
                for k in 0..latent {
                    a += lat[k] as f64 * w[k] as f64;
                }
                if o < g.nope {
                    kout[rr * g.nope + o] = a as f32;
                } else {
                    vout[rr * g.v_dim + (o - g.nope)] = a as f32;
                }
            }
            continue;
        }
        let mut e = 0;
        while e < per {
            let cnt = (per - e).min(4);
            let base = head * per + e;
            let mut acc = [0f32; 4];
            if cnt == 4 {
                let w0 = &wq[base * latent..(base + 1) * latent];
                let w1 = &wq[(base + 1) * latent..(base + 2) * latent];
                let w2 = &wq[(base + 2) * latent..(base + 3) * latent];
                let w3 = &wq[(base + 3) * latent..(base + 4) * latent];
                for k in 0..latent {
                    let xk = lat[k];
                    acc[0] = xk.mul_add(w0[k], acc[0]);
                    acc[1] = xk.mul_add(w1[k], acc[1]);
                    acc[2] = xk.mul_add(w2[k], acc[2]);
                    acc[3] = xk.mul_add(w3[k], acc[3]);
                }
            } else {
                for (j, a) in acc.iter_mut().enumerate().take(cnt) {
                    let w = &wq[(base + j) * latent..(base + j + 1) * latent];
                    for k in 0..latent {
                        *a = lat[k].mul_add(w[k], *a);
                    }
                }
            }
            for (j, a) in acc.iter().enumerate().take(cnt) {
                let o = e + j;
                let val = bf16r(*a);
                if o < g.nope {
                    kout[rr * g.nope + o] = val;
                } else {
                    vout[rr * g.v_dim + (o - g.nope)] = val;
                }
            }
            e += cnt;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn expand_rows_fma(
    x: &Expand<'_>,
    head: usize,
    r0: usize,
    kout: &mut [f32],
    vout: &mut [f32],
) {
    expand_rows_body(x, head, r0, kout, vout)
}

#[cfg(target_arch = "x86_64")]
fn expand_rows(x: &Expand<'_>, head: usize, r0: usize, kout: &mut [f32], vout: &mut [f32]) {
    if std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the running CPU reports FMA support.
        unsafe { expand_rows_fma(x, head, r0, kout, vout) }
    } else {
        expand_rows_body(x, head, r0, kout, vout)
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn expand_rows(x: &Expand<'_>, head: usize, r0: usize, kout: &mut [f32], vout: &mut [f32]) {
    expand_rows_body(x, head, r0, kout, vout)
}

/// Per-head expanded K_nope / V of every cache row: `[H][total][nope]`, `[H][total][v]`.
fn cpu_expand(x: &Expand<'_>, total: usize) -> (Vec<f32>, Vec<f32>) {
    const RB: usize = 64;
    let g = x.g;
    let mut kexp = vec![0f32; g.n_heads * total * g.nope];
    let mut vexp = vec![0f32; g.n_heads * total * g.v_dim];
    let mut tasks = Vec::new();
    for (hh, (kh, vh)) in kexp
        .chunks_mut(total * g.nope)
        .zip(vexp.chunks_mut(total * g.v_dim))
        .enumerate()
    {
        for (b, (kb, vb)) in kh
            .chunks_mut(RB * g.nope)
            .zip(vh.chunks_mut(RB * g.v_dim))
            .enumerate()
        {
            tasks.push((hh, b * RB, kb, vb));
        }
    }
    let queue = Mutex::new(tasks);
    std::thread::scope(|s| {
        for _ in 0..worker_count() {
            s.spawn(|| loop {
                let task = queue.lock().unwrap().pop();
                match task {
                    Some((hh, r0, kb, vb)) => expand_rows(x, hh, r0, kb, vb),
                    None => break,
                }
            });
        }
    });
    drop(queue);
    (kexp, vexp)
}

/// Independent CPU reference of one long case.
struct CpuRef {
    /// `[N, H*v]` BF16-valued F32.
    y: Vec<f32>,
    tasks: usize,
    /// Largest softmax probability over all (row, head) tasks.
    max_prob: f64,
    mean_max_prob: f64,
    /// (row, head) tasks whose argmax key is a planted needle row.
    peak_on_planted: usize,
    /// Max / mean over tasks of the softmax mass on the planted rows.
    planted_mass_max: f64,
    planted_mass_mean: f64,
    /// (row, head) tasks with more than 0.3 softmax mass on the planted rows.
    planted_mass_gt_0_3: usize,
}

/// `wq`: BF16-rounded expansion weight `[H*(nope+v), latent]`. Scores
/// `scale * dot` in f64, causal `j <= P + i`, softmax in f64, `y = bf16(sum p*v)`.
fn cpu_reference(
    g: &LatentGeometry,
    p: usize,
    n: usize,
    d: &LongData,
    wq: &[f32],
    exact: bool,
) -> CpuRef {
    let (h, nope, tail, v, latent, width, dqk) = (
        g.n_heads,
        g.nope,
        g.tail,
        g.v_dim,
        g.latent,
        g.row_width(),
        g.qk_dim(),
    );
    let total = p + n;
    let (kexp, vexp) = cpu_expand(
        &Expand {
            g,
            cache: &d.cache,
            wq,
            round_kv: !exact,
        },
        total,
    );
    let scale = (dqk as f64).powf(-0.5);
    let tasks_n = n * h;
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, Vec<f32>, f64, bool, f64)>> =
        Mutex::new(Vec::with_capacity(tasks_n));
    std::thread::scope(|s| {
        for _ in 0..worker_count().min(tasks_n) {
            s.spawn(|| {
                let mut scores: Vec<f64> = Vec::new();
                let mut acc = vec![0f64; v];
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= tasks_n {
                        break;
                    }
                    let (i, hh) = (idx / h, idx % h);
                    let len = p + i + 1;
                    let qv = &d.q[(i * h + hh) * dqk..(i * h + hh + 1) * dqk];
                    let kh = &kexp[hh * total * nope..(hh + 1) * total * nope];
                    let vh = &vexp[hh * total * v..(hh + 1) * total * v];
                    scores.clear();
                    let mut best = (f64::NEG_INFINITY, 0usize);
                    for j in 0..len {
                        let kn = &kh[j * nope..(j + 1) * nope];
                        let tl = &d.cache[j * width + latent..(j + 1) * width];
                        let mut dot = 0f64;
                        for c in 0..nope {
                            dot += qv[c] as f64 * kn[c] as f64;
                        }
                        for c in 0..tail {
                            dot += qv[nope + c] as f64 * tl[c] as f64;
                        }
                        let sc = scale * dot;
                        if sc > best.0 {
                            best = (sc, j);
                        }
                        scores.push(sc);
                    }
                    let mut sum = 0f64;
                    acc.fill(0.0);
                    for (j, sc) in scores.iter().enumerate() {
                        let w = (sc - best.0).exp();
                        sum += w;
                        let vr = &vh[j * v..(j + 1) * v];
                        for c in 0..v {
                            acc[c] += w * vr[c] as f64;
                        }
                    }
                    let y: Vec<f32> = acc
                        .iter()
                        .map(|a| {
                            let y = (a / sum) as f32;
                            if exact {
                                y
                            } else {
                                bf16r(y)
                            }
                        })
                        .collect();
                    let planted_mass = d
                        .planted
                        .iter()
                        .map(|&j| (scores[j] - best.0).exp())
                        .sum::<f64>()
                        / sum;
                    results.lock().unwrap().push((
                        idx,
                        y,
                        1.0 / sum,
                        d.planted.contains(&best.1),
                        planted_mass,
                    ));
                }
            });
        }
    });
    let results = results.into_inner().unwrap();
    let mut y = vec![0f32; n * h * v];
    let (mut max_prob, mut sum_prob, mut peak_on_planted) = (0f64, 0f64, 0usize);
    let (mut pm_max, mut pm_sum, mut pm_gt) = (0f64, 0f64, 0usize);
    for (idx, yv, pmax, on_planted, pm) in &results {
        y[idx * v..(idx + 1) * v].copy_from_slice(yv);
        max_prob = max_prob.max(*pmax);
        sum_prob += pmax;
        peak_on_planted += *on_planted as usize;
        pm_max = pm_max.max(*pm);
        pm_sum += pm;
        pm_gt += (*pm > 0.3) as usize;
    }
    let cnt = results.len().max(1) as f64;
    CpuRef {
        y,
        tasks: results.len(),
        max_prob,
        mean_max_prob: sum_prob / cnt,
        peak_on_planted,
        planted_mass_max: pm_max,
        planted_mass_mean: pm_sum / cnt,
        planted_mass_gt_0_3: pm_gt,
    }
}

/// BF16-rounded `kv_b` of the real checkpoint weights file (loaded once).
fn long_weights<'a>(dir: &Path, slot: &'a mut Option<Vec<f32>>) -> Result<&'a [f32], String> {
    if slot.is_none() {
        let g = RAON_GEOMETRY;
        let t = Tensors::load(&dir.join(LONG_WEIGHTS))?;
        let kv_b = t.get("kv_b", g.expanded_width() * g.latent)?;
        *slot = Some(kv_b.iter().map(|f| bf16r(*f)).collect());
    }
    slot.as_deref()
        .ok_or_else(|| "long-case weights missing".to_string())
}

fn run_long_case(
    gpu: &mut Gpu,
    kv_b: &[f32],
    p: usize,
    n: usize,
    routes: Routes,
) -> Result<Value, String> {
    let name = format!("long_p{p}_n{n}");
    let g = RAON_GEOMETRY;
    let (h, v, per) = (g.n_heads, g.v_dim, g.nope + g.v_dim);
    let (width, dqk) = (g.row_width(), g.qk_dim());
    let total = p + n;
    let cap = total + 37; // latent-cache rows (poison beyond the live length)
    let mcap = total + 11; // mirror rows
    let scale = g.default_scale();

    let data = gen_long(&g, p, n);
    let t0 = Instant::now();
    let cpu = cpu_reference(&g, p, n, &data, kv_b, false);
    // Unrounded-f64 reference: what the absorbed route computes.
    let cpu_exact = routes
        .absorbed
        .then(|| cpu_reference(&g, p, n, &data, kv_b, true));
    eprintln!(
        "mla {name}: CPU reference done in {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    // Device state: poisoned cache/mirrors; all P+N rows uploaded directly.
    let cache_t = poison_bf16(gpu, &[cap, width])?;
    let bytes: Vec<u8> = data
        .cache
        .iter()
        .flat_map(|f| f32_to_bf16(*f).to_le_bytes())
        .collect();
    gpu.memcpy_htod_auto(&cache_t.buf, &bytes).map_err(he)?;
    drop(bytes);
    let knope = poison_bf16(gpu, &[mcap, h, g.nope])?;
    let vmir = poison_bf16(gpu, &[mcap, h, v])?;
    let pos = gpu.alloc_tensor(&[1], DType::F32).map_err(he)?;
    set_pos(gpu, &pos, p)?;
    let d_q = up_f32(gpu, &data.q, &[n, h * dqk])?;
    let d_w = up_bf16(gpu, kv_b, &[h * per, g.latent])?;
    let d_yr = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let d_yt1 = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let d_yt2 = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let d_ya = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let rd = RouteDev::new(gpu, &g, n, routes, kv_b)?;

    let outcome = (|| -> Result<Vec<Cmp>, String> {
        let cache = LatentCache {
            rows: &cache_t,
            capacity_rows: cap,
            pos: &pos,
            host_pos: Some(p),
        };
        let mirror = ExpandedMirror {
            knope: &knope,
            vmir: &vmir,
            capacity_rows: mcap,
            q8: false,
        };
        let dense = KvExpansion::Dense { weight: &d_w };
        // Production Decompress over absolute row tiles (all P+N rows).
        let mut s = 0;
        while s < total {
            let c = LONG_TILE.min(total - s);
            execute_latent_attention(
                gpu,
                &LatentAttentionOp::Decompress {
                    geometry: g,
                    cache,
                    mirror,
                    expansion: dense,
                    range: RowRange::Absolute { start: s, count: c },
                },
            )
            .map_err(he)?;
            s += c;
        }
        attn(
            gpu,
            &g,
            cache,
            Some(mirror),
            &d_q,
            &d_yr,
            n,
            scale,
            AttentionRoute::Reference,
        )?;
        if routes.tiled {
            for y in [&d_yt1, &d_yt2] {
                attn(gpu, &g, cache, Some(mirror), &d_q, y, n, scale, rd.tiled())?;
            }
        }
        if routes.absorbed {
            attn(gpu, &g, cache, None, &d_q, &d_ya, n, scale, rd.absorbed()?)?;
        }
        gpu.hip.device_synchronize().map_err(he)?;
        let yr = gpu.download_f32(&d_yr).map_err(he)?;
        // Q8 mirror: quantize the dense mirror just built, run tiled on it and
        // Reference on its dequantization.
        let q8 = if routes.tiled {
            Some(q8_mirror_attention(
                gpu, &g, cache, &rd, &knope, &vmir, total, mcap, &d_q, n, scale,
            )?)
        } else {
            None
        };

        let lab = |c: Cmp, route: &str, bit_exact: bool| {
            let mut c = labeled(c, route, bit_exact);
            c.json["P"] = json!(p);
            c.json["N"] = json!(n);
            c
        };
        let mut checks: Vec<Cmp> = Vec::new();
        let needle_ok = cpu.planted_mass_gt_0_3 > 0;
        checks.push(lab(
            Cmp {
                ok: needle_ok,
                json: json!({
                    "tensor": "needle_peak_present", "pass": needle_ok,
                    "tasks_with_planted_mass_gt_0_3": cpu.planted_mass_gt_0_3,
                    "peak_on_planted": cpu.peak_on_planted, "tasks": cpu.tasks,
                }),
            },
            "cpu",
            false,
        ));
        checks.push(lab(
            compare("attn_y_reference_vs_cpu", &yr, &cpu.y),
            "reference",
            false,
        ));
        let mut yt1 = Vec::new();
        if routes.tiled {
            yt1 = gpu.download_f32(&d_yt1).map_err(he)?;
            let yt2 = gpu.download_f32(&d_yt2).map_err(he)?;
            checks.push(lab(
                compare("attn_y_tiled_vs_cpu", &yt1, &cpu.y),
                "tiled",
                false,
            ));
            checks.push(lab(
                exact(
                    "attn_y_tiled_repeat_bytewise",
                    &f32_bytes(&yt1),
                    &f32_bytes(&yt2),
                ),
                "tiled",
                true,
            ));
            checks.push(lab(
                compare("reference_vs_tiled", &yt1, &yr),
                "tiled",
                false,
            ));
        }
        if routes.absorbed {
            let ya = gpu.download_f32(&d_ya).map_err(he)?;
            let exact_y = cpu_exact.as_ref().map(|c| &c.y[..]).unwrap_or(&[]);
            checks.push(lab(
                compare("attn_y_absorbed_vs_cpu", &ya, exact_y),
                "absorbed",
                false,
            ));
            checks.push(informational(lab(
                compare("attn_y_absorbed_vs_cpu_bf16kv", &ya, &cpu.y),
                "absorbed",
                false,
            )));
            if routes.tiled {
                checks.push(lab(
                    band_no_outlier(compare("absorbed_vs_tiled", &ya, &yt1)),
                    "absorbed",
                    false,
                ));
            }
        }
        if let Some(q8) = &q8 {
            checks.push(lab(
                compare_in("attn_y_tiled_q8_vs_cpu", &q8.y1, &cpu.y, Q8_TRUTH_BAND),
                "tiled_q8",
                false,
            ));
            checks.push(lab(
                exact(
                    "attn_y_tiled_q8_repeat_bytewise",
                    &f32_bytes(&q8.y1),
                    &f32_bytes(&q8.y2),
                ),
                "tiled_q8",
                true,
            ));
            checks.push(lab(
                compare_in(
                    "attn_y_tiled_q8_vs_q8_reference",
                    &q8.y1,
                    &q8.yref,
                    Q8_REF_BAND,
                ),
                "tiled_q8",
                false,
            ));
        }
        Ok(checks)
    })();

    rd.free(gpu);
    for t in [
        cache_t, knope, vmir, pos, d_q, d_w, d_yr, d_yt1, d_yt2, d_ya,
    ] {
        let _ = gpu.free_tensor(t);
    }
    let checks = outcome?;

    let (pass, check_json) = finish_checks(checks);
    Ok(json!({
        "name": name, "pass": pass, "P": p, "N": n,
        "dims": {"N": n, "P": p, "H": h, "nope": g.nope, "tail": g.tail, "v": v,
                 "latent": g.latent},
        "generated": {
            "prng": "xorshift64* + Box-Muller", "base_seed": format!("0x{LONG_SEED:016x}"),
            "case_seed": format!("0x{:016x}", long_seed(p, n)),
            "case_seed_rule": "base_seed ^ (P << 20) ^ N",
            "latent_tail_std": 1.0, "query_std": LONG_QUERY_STD,
            "needle_gain": LONG_NEEDLE_GAIN, "planted_rows": data.planted,
            "planted_tail": "gain * mean over heads of the tail of query row N-1",
            "weights": LONG_WEIGHTS,
        },
        "cpu_reference": {
            "tasks": cpu.tasks, "max_prob": cpu.max_prob, "mean_max_prob": cpu.mean_max_prob,
            "peak_on_planted": cpu.peak_on_planted,
            "planted_mass_max": cpu.planted_mass_max,
            "planted_mass_mean": cpu.planted_mass_mean,
            "planted_mass_gt_0_3_tasks": cpu.planted_mass_gt_0_3,
        },
        "routes": routes.names(),
        "checks": check_json,
    }))
}

/// `--long 0` skips the generated long-context cases.
fn long_enabled(opts: &BTreeMap<String, String>) -> Result<bool, String> {
    match opts.get("long") {
        None => Ok(true),
        Some(s) => s
            .parse::<usize>()
            .map(|v| v != 0)
            .map_err(|e| format!("--long `{s}`: {e}")),
    }
}

/// Entry point used by `raon_oracle module --module mla`.
pub fn run(fixtures: &Path, opts: &BTreeMap<String, String>) -> Result<Value, String> {
    let dir = fixtures.join("mla");
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(dir.join("manifest.json"))
            .map_err(|e| format!("{}: {e}", dir.join("manifest.json").display()))?,
    )
    .map_err(|e| format!("mla manifest.json: {e}"))?;
    let cases = manifest["cases"]
        .as_array()
        .ok_or("manifest.cases missing")?;
    if cases.is_empty() {
        return Err("mla manifest has no cases".into());
    }
    let routes = Routes::parse(opts)?;
    let long_on = long_enabled(opts)? && !opts.contains_key("negative");
    let filter = opts.get("case");
    let mut gpu = Gpu::init().map_err(he)?;
    let mut out = Vec::new();
    let mut all = true;
    let mut ran = 0usize;
    for case in cases {
        let nm = case["name"].as_str().unwrap_or("");
        if filter.is_some_and(|f| !nm.contains(f.as_str())) {
            continue;
        }
        ran += 1;
        let r = match run_case(&mut gpu, &dir, case, routes) {
            Ok(v) => v,
            Err(e) => json!({"name": nm, "pass": false, "error": e}),
        };
        all &= r["pass"].as_bool().unwrap_or(false);
        out.push(r);
    }
    let mut long_ran = 0usize;
    if long_on {
        let mut weights: Option<Vec<f32>> = None;
        for (p, n) in LONG_CASES {
            let nm = format!("long_p{p}_n{n}");
            if filter.is_some_and(|f| !nm.contains(f.as_str())) {
                continue;
            }
            ran += 1;
            long_ran += 1;
            let r = match long_weights(&dir, &mut weights)
                .and_then(|w| run_long_case(&mut gpu, w, p, n, routes))
            {
                Ok(v) => v,
                Err(e) => json!({"name": nm, "pass": false, "P": p, "N": n, "error": e}),
            };
            all &= r["pass"].as_bool().unwrap_or(false);
            out.push(r);
        }
    }
    if ran == 0 {
        return Err(format!("no mla case matched filter {filter:?}"));
    }
    Ok(json!({
        "module": "mla", "pass": all, "cases_run": ran, "long_cases_run": long_ran,
        "routes": routes.names(),
        "ref_sha256": manifest.get("ref_sha256").cloned().unwrap_or(Value::Null),
        "cases": out,
    }))
}
