// SPDX-License-Identifier: Apache-2.0
//! Timing of the EP owner-routed symmetric MoE prefill chain: A4 (`qwen4_moe_rotate256_i4` ->
//! `gemm_qwen4_moe_gate_up_silu_iu4_sym` -> `qwen4_moe_rotate128_i4_gp` -> `gemm_qwen4_moe_down_iu4_sym`) against
//! the A8 twin (`qwen4_moe_rotate256_i8` -> `gemm_qwen4_moe_gate_up_silu_a8_sym` -> `qwen4_moe_rotate128_i8` ->
//! `gemm_qwen4_moe_down_a8_sym`), real Flash-Next expert tensors of layers rotated per iteration (cold against the
//! MALL), the production owner mask (EP2 rank 0: experts `e % 2 == 0` owned, the rest sentinel tiles), uniform
//! top-10 routing over 512 experts. Run ignored on fabric card C with `--class timing`.
//!
//! Env: `EP_A8_MODEL` (.mq4-xts), `EP_A8_OUT` (dir for the JSON), `EP_A8_PLANAR` (`1`: F1 drop-zero planar
//! experts, the production EP2 layout; default `0` shipped), `EP_A8_ROWS` (default `9,64,256,512,2048`),
//! `EP_A8_LAYERS` (default `0,17,47`), `EP_A8_REPS` (default 5), `HIPFIRE_QWEN4_MOE_EP_A8_NT` (tile-run width).
use rdna_compute::{DType, Gpu, GpuTensor};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    time::Instant,
};

const DEFAULT_MODEL: &str =
    "/home/kaden/qcal/release-0.4.3/mgfn/model/qwen3.8-flash-next-base.mq4-xts";
const E: usize = 512;
const H: usize = 2560;
const I: usize = 640;
const TOPK: usize = 10;
const GU_BYTES: usize = 1_740_800;
const DN_BYTES: usize = 870_400;
type Result<T, Error = Box<dyn std::error::Error>> = std::result::Result<T, Error>;

struct Tensor {
    name: String,
    offset: u64,
    len: usize,
}

fn index(f: &mut File) -> Result<Vec<Tensor>> {
    let mut h = [0; 32];
    f.read_exact(&mut h)?;
    assert_eq!(&h[..4], b"HFQM");
    let u32_at = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap()) as usize;
    let u64_at = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap());
    let meta = u64_at(&h[16..24]);
    let data = u64_at(&h[24..32]);
    f.seek(SeekFrom::Start(meta))?;
    let mut b = vec![0; (data - meta) as usize];
    f.read_exact(&mut b)?;
    let (mut depth, mut quoted, mut escape, mut p) = (0i32, false, false, 0);
    for (i, &c) in b.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if quoted && c == b'\\' {
            escape = true;
            continue;
        }
        if c == b'"' {
            quoted = !quoted;
        }
        if !quoted {
            if c == b'{' {
                depth += 1;
            }
            if c == b'}' {
                depth -= 1;
                if depth == 0 {
                    p = i + 1;
                    break;
                }
            }
        }
    }
    assert!(p > 0);
    let n = u32_at(&b[p..p + 4]);
    p += 4;
    assert_eq!(n, u32_at(&h[12..16]));
    let mut offset = data;
    let mut out = Vec::new();
    for _ in 0..n {
        let l = u16::from_le_bytes(b[p..p + 2].try_into().unwrap()) as usize;
        p += 2;
        let name = String::from_utf8(b[p..p + l].to_vec())?;
        p += l + 1;
        let nd = b[p] as usize;
        p += 1;
        p += 4 * nd;
        p += 4;
        let len = u64_at(&b[p..p + 8]) as usize;
        p += 8;
        out.push(Tensor { name, offset, len });
        offset += len as u64;
    }
    Ok(out)
}

fn payload(f: &mut File, t: &Tensor) -> Result<Vec<u8>> {
    f.seek(SeekFrom::Start(t.offset))?;
    let mut b = vec![0; t.len];
    f.read_exact(&mut b)?;
    Ok(b)
}

/// One shipped expert (`rows x k`, groups of `group_cols` columns, `group_bytes` per group: `scales` f16 header
/// words then the nibble bytes) -> planar `[codes rows*k/2][scales rows*(k/128) f16]`.
fn repack_expert(src: &[u8], rows: usize, k: usize, gate_up: bool) -> Vec<u8> {
    let (group_cols, group_bytes, hdr) = if gate_up { (256, 136, 8) } else { (128, 68, 4) };
    let groups = k / group_cols;
    let halves = group_cols / 128;
    let mut codes = Vec::with_capacity(rows * k / 2);
    let mut scales = Vec::with_capacity(rows * (k / 128) * 2);
    for r in 0..rows {
        for g in 0..groups {
            let base = (r * groups + g) * group_bytes;
            codes.extend_from_slice(&src[base + hdr..base + group_bytes]);
            for h in 0..halves {
                // header words: [sc_h, zp_h] per half
                scales.extend_from_slice(&src[base + 4 * h..base + 4 * h + 2]);
            }
        }
    }
    codes.extend_from_slice(&scales);
    codes
}

fn raw(g: &mut Gpu, n: usize) -> Result<GpuTensor> {
    Ok(g.alloc_tensor(&[n.div_ceil(4)], DType::F32)?)
}

fn ints(v: impl Iterator<Item = i32>) -> Vec<u8> {
    v.flat_map(i32::to_le_bytes).collect()
}

fn env_list(name: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

struct WeightSet {
    layer: usize,
    gu: GpuTensor,
    dn: GpuTensor,
    gu_ptrs: GpuTensor,
    dn_ptrs: GpuTensor,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "real FN artifact, gfx1201 fabric card C, --class timing"]
fn qwen4_moe_ep_a8_chain_timing() -> Result<()> {
    let model = std::env::var("EP_A8_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
    let out_dir = std::env::var("EP_A8_OUT").unwrap_or_else(|_| ".".into());
    let planar = std::env::var("EP_A8_PLANAR")
        .map(|v| v == "1")
        .unwrap_or(false);
    let rows_list = env_list("EP_A8_ROWS", &[9, 64, 256, 512, 2048]);
    let layers = env_list("EP_A8_LAYERS", &[0, 17, 47]);
    let reps = env_list("EP_A8_REPS", &[5])[0];
    let mut g = Gpu::init()?;
    assert_eq!(g.arch, "gfx1201", "timing needs exact gfx1201");
    g.set_mq4_experts_planar(planar);
    let (gu_stride, dn_stride) = if planar {
        (1_689_600usize, 844_800usize)
    } else {
        (GU_BYTES, DN_BYTES)
    };
    let mut f = File::open(&model)?;
    let ts = index(&mut f)?;
    let mut sets = Vec::new();
    for &layer in &layers {
        let find = |suffix: &str| {
            ts.iter()
                .find(|t| {
                    t.name.starts_with("model.language_model.")
                        && t.name.contains(&format!("layers.{layer}.mlp.experts."))
                        && t.name.ends_with(suffix)
                })
                .unwrap_or_else(|| panic!("tensor layers.{layer} {suffix}"))
        };
        let (gu_t, dn_t) = (find("experts.gate_up_proj"), find("experts.down_proj"));
        assert_eq!((gu_t.len, dn_t.len), (E * GU_BYTES, E * DN_BYTES));
        let gu_b = payload(&mut f, gu_t)?;
        let dn_b = payload(&mut f, dn_t)?;
        let (gu_b, dn_b) = if planar {
            let mut gp = Vec::with_capacity(E * gu_stride);
            let mut dp = Vec::with_capacity(E * dn_stride);
            for e in 0..E {
                gp.extend(repack_expert(
                    &gu_b[e * GU_BYTES..(e + 1) * GU_BYTES],
                    2 * I,
                    H,
                    true,
                ));
                dp.extend(repack_expert(
                    &dn_b[e * DN_BYTES..(e + 1) * DN_BYTES],
                    H,
                    I,
                    false,
                ));
            }
            (gp, dp)
        } else {
            (gu_b, dn_b)
        };
        let gu = g.upload_raw(&gu_b, &[gu_b.len()])?;
        let dn = g.upload_raw(&dn_b, &[dn_b.len()])?;
        drop((gu_b, dn_b));
        let table = |w: &GpuTensor, stride: usize| {
            (0..E)
                .flat_map(|e| ((w.buf.as_ptr() as u64) + (e * stride) as u64).to_le_bytes())
                .collect::<Vec<_>>()
        };
        let gu_ptrs = g.upload_raw(&table(&gu, gu_stride), &[E * 8])?;
        let dn_ptrs = g.upload_raw(&table(&dn, dn_stride), &[E * 8])?;
        sets.push(WeightSet {
            layer,
            gu,
            dn,
            gu_ptrs,
            dn_ptrs,
        });
        eprintln!("loaded layer {layer} (planar={planar})");
    }
    let per_expert = gu_stride + dn_stride;
    let mut results = Vec::new();
    for &n in &rows_list {
        let slots = n * TOPK;
        let p = (slots + E * 15).div_ceil(16) * 16;
        g.reserve_qwen4_moe_sym(n, TOPK, E, H, I)?;
        // Deterministic uniform top-10 (distinct experts per token).
        let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ (n as u64);
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut topk = Vec::with_capacity(slots);
        for _ in 0..n {
            let mut chosen = Vec::new();
            while chosen.len() < TOPK {
                let e = (next() % E as u64) as i32;
                if !chosen.contains(&e) {
                    chosen.push(e);
                }
            }
            topk.extend(chosen);
        }
        let top = g.upload_raw(&ints(topk.iter().copied()), &[slots])?;
        let xs: Vec<f32> = (0..n * H)
            .map(|i| {
                let v = ((i as u64).wrapping_mul(2654435761) >> 7) as u32 % 4001;
                let base = (v as f32 - 2000.0) / 1000.0;
                if i % H % 97 == 0 {
                    base * 8.0
                } else {
                    base
                }
            })
            .collect();
        let x = g.upload_f32(&xs, &[n * H])?;
        let counts = raw(&mut g, E * 4)?;
        let offsets = raw(&mut g, (E + 1) * 4)?;
        let sorted = raw(&mut g, p * 4)?;
        let tiles = raw(&mut g, p / 16 * 4)?;
        let inverse = raw(&mut g, slots * 4)?;
        g.moe_scatter_stable_top10(
            &top, &counts, &offsets, &sorted, &tiles, &inverse, slots, E, p,
        )?;
        let owner = g.upload_raw(&ints((0..E).map(|e| (e % 2) as i32)), &[E])?;
        let owned = raw(&mut g, p / 16 * 4)?;
        g.ep_tile_mask(&tiles, &owned, &owner, p / 16, E, 0)?;
        g.hip.device_synchronize()?;
        let mut tile_ids = vec![0u8; p / 16 * 4];
        g.hip.memcpy_dtoh(&mut tile_ids, &owned.buf)?;
        let tile_ids: Vec<i32> = tile_ids
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let owned_tiles = tile_ids.iter().filter(|&&e| e >= 0).count();
        let mut active: Vec<i32> = tile_ids.iter().copied().filter(|&e| e >= 0).collect();
        active.sort_unstable();
        active.dedup();
        let weights_bytes = active.len() * per_expert;
        let ygu = raw(&mut g, p * I * 2)?;
        let ydn = raw(&mut g, p * H * 2)?;
        let iters = match n {
            0..=16 => 120,
            17..=128 => 80,
            129..=300 => 40,
            301..=700 => 24,
            _ => 10,
        };
        // arm 0 = A4, 1 = A8; stage 0 = full chain, 1 = gate stage (producer + gate/up), 2 = down stage.
        let mut run = |g: &mut Gpu, arm: usize, stage: usize, w: &WeightSet| -> Result<()> {
            if arm == 0 {
                if stage != 2 {
                    let a = g.qwen4_moe_rotate256_i4(&x, H, n)?;
                    g.gemm_qwen4_moe_gate_up_silu_iu4_sym(
                        &w.gu_ptrs,
                        &owned,
                        &sorted,
                        &a,
                        &ygu,
                        2 * I,
                        H,
                        TOPK,
                        p,
                        n,
                        false,
                    )?;
                }
                if stage != 1 {
                    let d = g.qwen4_moe_rotate128_i4_gp(&ygu, &sorted, I, p, slots)?;
                    g.gemm_qwen4_moe_down_iu4_sym(
                        &w.dn_ptrs, &owned, &sorted, &d, &ydn, H, I, 1, p, slots, false,
                    )?;
                }
            } else {
                if stage != 2 {
                    let a = g.qwen4_moe_rotate256_i8(&x, H, n)?;
                    g.gemm_qwen4_moe_gate_up_silu_a8_sym(
                        &w.gu_ptrs,
                        &owned,
                        &sorted,
                        &a,
                        &ygu,
                        2 * I,
                        H,
                        TOPK,
                        p,
                        n,
                    )?;
                }
                if stage != 1 {
                    let d = g.qwen4_moe_rotate128_i8(&ygu, &sorted, I, p, slots)?;
                    g.gemm_qwen4_moe_down_a8_sym(
                        &w.dn_ptrs, &owned, &sorted, &d, &ydn, H, I, 1, p, slots,
                    )?;
                }
            }
            Ok(())
        };
        let mut samples: [[Vec<f64>; 3]; 2] = Default::default();
        // Warm every kernel, every weight set.
        for arm in 0..2 {
            for stage in 0..3 {
                for w in &sets {
                    run(&mut g, arm, stage, w)?;
                }
            }
        }
        g.hip.device_synchronize()?;
        for rep in 0..reps {
            for stage in 0..3 {
                let order: [usize; 2] = if (rep + stage) % 2 == 0 {
                    [0, 1]
                } else {
                    [1, 0]
                };
                for arm in order {
                    g.hip.device_synchronize()?;
                    let t0 = Instant::now();
                    for it in 0..iters {
                        run(&mut g, arm, stage, &sets[it % sets.len()])?;
                    }
                    g.hip.device_synchronize()?;
                    samples[arm][stage].push(t0.elapsed().as_secs_f64() * 1e6 / iters as f64);
                }
            }
        }
        let mut row = serde_like(n, p, owned_tiles, active.len(), weights_bytes);
        for (arm, name) in ["a4", "a8"].iter().enumerate() {
            for (stage, sname) in ["chain", "gate", "down"].iter().enumerate() {
                let v = &mut samples[arm][stage];
                let min = v.iter().cloned().fold(f64::MAX, f64::min);
                let med = median(v);
                row.push_str(&format!(
                    ",\"{name}_{sname}_us_median\":{med:.2},\"{name}_{sname}_us_min\":{min:.2},\"{name}_{sname}_samples\":{:?}",
                    v
                ));
                if stage == 0 {
                    row.push_str(&format!(
                        ",\"{name}_weights_only_gbps\":{:.1}",
                        weights_bytes as f64 / (med * 1e-6) / 1e9
                    ));
                }
            }
        }
        row.push('}');
        println!("EP_A8_TIMING {row}");
        results.push(row);
        for t in [
            top, x, counts, offsets, sorted, tiles, inverse, owner, owned, ygu, ydn,
        ] {
            g.free_tensor(t)?;
        }
    }
    let nt = std::env::var("HIPFIRE_QWEN4_MOE_EP_A8_NT").unwrap_or_else(|_| "auto".into());
    let json = format!(
        "{{\"schema\":\"qwen4-moe-ep-a8-time-v1\",\"model\":\"{model}\",\"planar\":{planar},\"layers\":{layers:?},\"nt\":\"{nt}\",\"reps\":{reps},\"routing\":\"uniform top-10 over 512 experts, EP2 rank-0 owner mask e%2\",\"activations\":\"synthetic F32 with outlier channels\",\"rows\":[{}]}}",
        results.join(",")
    );
    let path = format!(
        "{out_dir}/ep_a8_timing_planar{}_nt{nt}_pid{}.json",
        planar as u8,
        std::process::id()
    );
    std::fs::write(&path, &json)?;
    println!("EP_A8_TIMING_JSON {path}");
    let _ = sets.iter().map(|s| (s.layer, &s.gu, &s.dn)).count();
    Ok(())
}

fn serde_like(
    n: usize,
    p: usize,
    owned_tiles: usize,
    active: usize,
    weights_bytes: usize,
) -> String {
    format!(
        "{{\"rows\":{n},\"grouped_rows\":{p},\"owned_tiles\":{owned_tiles},\"active_owned_experts\":{active},\"weights_bytes\":{weights_bytes}"
    )
}
