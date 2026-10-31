// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon converter tests: class map, MXFP8 dequant, cross-shard pairing, padding,
//! symmetric headers, end-to-end synthetic checkpoint, final-code import.

use super::*;
use crate::quant_fwht::cpu_fwht_256;
use half::bf16;
use safetensors::tensor::{serialize_to_file, TensorView};
use safetensors::Dtype;
use std::collections::HashMap;
use std::path::PathBuf;

// ─── helpers ────────────────────────────────────────────────────────────────

fn vals(seed: u32, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| bf16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}

#[derive(Clone)]
struct T {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
    shard: Option<usize>,
}

fn bf(name: &str, shape: &[usize], seed: u32, scale: f32) -> T {
    let n: usize = shape.iter().product();
    T {
        name: name.into(),
        dtype: Dtype::BF16,
        shape: shape.to_vec(),
        data: bf16_bytes(&vals(seed, n, scale)),
        shard: None,
    }
}

fn f32t(name: &str, shape: &[usize], seed: u32) -> T {
    let n: usize = shape.iter().product();
    T {
        name: name.into(),
        dtype: Dtype::F32,
        shape: shape.to_vec(),
        data: vals(seed, n, 1.0)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        shard: None,
    }
}

fn gamma(name: &str, n: usize) -> T {
    // 1 + k/64 is exactly representable in BF16 and F16.
    let v: Vec<f32> = (0..n).map(|i| 1.0 + (i % 16) as f32 / 64.0).collect();
    T {
        name: name.into(),
        dtype: Dtype::BF16,
        shape: vec![n],
        data: bf16_bytes(&v),
        shard: None,
    }
}

fn fp8_pair(stem: &str, n: usize, k: usize, seed: u32) -> [T; 2] {
    let mut s = seed.wrapping_mul(2654435761);
    let w: Vec<u8> = (0..n * k)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let b = (s >> 16) as u8;
            if b & 0x7f == 0x7f {
                0x38
            } else {
                b
            }
        })
        .collect();
    let sc: Vec<u8> = (0..n * k / 32)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            120 + ((s >> 16) % 14) as u8
        })
        .collect();
    [
        T {
            name: format!("{stem}.weight"),
            dtype: Dtype::F8_E4M3,
            shape: vec![n, k],
            data: w,
            shard: None,
        },
        T {
            name: format!("{stem}.weight_scale_inv"),
            dtype: Dtype::U8,
            shape: vec![n, k / 32],
            data: sc,
            shard: None,
        },
    ]
}

const H: usize = 256;
const NEXP: usize = 2;

fn test_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "raon1_mini",
        "architectures": ["Raon1MiniForCausalLM"],
        "hidden_size": H,
        "num_hidden_layers": 3,
        "num_experts": NEXP,
        "first_k_dense_replace": 1,
        "num_shared_experts": 1,
        "moe_intermediate_size": 1088,
        "quantization_config": {
            "quant_method": "mxfp8", "activation_scheme": "dynamic",
            "weight_block_size": [1, 32], "dequantize": false
        }
    })
}

fn mhc(p: &str, site: &str, t: &mut Vec<T>, seed: u32) {
    let q = format!("{p}{site}.");
    for (n, shape) in [
        ("alpha_post", vec![]),
        ("alpha_pre", vec![]),
        ("b_post", vec![4]),
        ("b_pre", vec![4]),
    ] {
        t.push(gamma(
            &format!("{q}{n}"),
            shape.iter().product::<usize>().max(1),
        ));
        t.last_mut().unwrap().shape = shape;
    }
    t.push(bf(&format!("{q}phi_post"), &[4, 1024], seed, 0.5));
    t.push(bf(&format!("{q}phi_pre"), &[4, 1024], seed + 1, 0.5));
    t.push(gamma(&format!("{q}res.alpha_z"), 1));
    t.last_mut().unwrap().shape = vec![];
    t.push(gamma(&format!("{q}res.b_z"), 9));
    t.push(bf(&format!("{q}res.phi_z"), &[9, 1024], seed + 2, 0.5));
    t.push(f32t(&format!("{q}res.eta"), &[], seed + 3));
}

/// 3 layers: 0 = RDA + dense MLP, 1 = RDA + MoE, 2 = MLA + MoE; hidden 256.
fn synthetic_tensors() -> Vec<T> {
    let mut t = vec![
        bf("model.embed_tokens.weight", &[64, H], 1, 0.1),
        bf("lm_head.weight", &[64, H], 2, 0.1),
        gamma("model.norm.weight", H),
    ];
    for l in 0..3usize {
        let p = format!("model.layers.{l}.");
        let s = 100 * (l as u32 + 1);
        t.push(gamma(&format!("{p}input_layernorm.weight"), H));
        t.push(gamma(&format!("{p}post_attention_layernorm.weight"), H));
        mhc(&p, "self_attn_raon_mhc", &mut t, s);
        mhc(&p, "mlp_raon_mhc", &mut t, s + 10);
        let a = format!("{p}self_attn.");
        if l < 2 {
            for (n, sh) in [
                ("q_proj", [512, H]),
                ("k_proj", [512, H]),
                ("v_proj", [512, H]),
                ("g_proj", [512, H]),
                ("o_proj", [H, 512]),
                ("f_a_proj", [32, H]),
                ("f_b_proj", [512, 32]),
                ("b_proj", [4, H]),
                ("rda_w_proj", [4, H]),
            ] {
                t.push(bf(&format!("{a}{n}.weight"), &sh, s + 20, 0.2));
            }
            t.push(f32t(&format!("{a}A_log"), &[4], s + 30));
            t.push(f32t(&format!("{a}dt_bias"), &[512], s + 31));
            for n in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                t.push(bf(&format!("{a}{n}.weight"), &[512, 4], s + 32, 0.5));
            }
            t.push(gamma(&format!("{a}o_norm.weight"), 128));
        } else {
            for (n, sh) in [
                ("q_a_proj", [96, H]),
                ("q_b_proj", [160, 96]),
                ("kv_a_proj_with_mqa", [72, H]),
                ("kv_b_proj", [160, 256]),
                ("g_proj", [512, H]),
                ("o_proj", [H, 512]),
            ] {
                t.push(bf(&format!("{a}{n}.weight"), &sh, s + 40, 0.2));
            }
            t.push(gamma(&format!("{a}q_a_layernorm.weight"), 96));
            t.push(gamma(&format!("{a}kv_a_layernorm.weight"), 256));
        }
        if l == 0 {
            t.push(bf(
                &format!("{p}mlp.gate_proj.weight"),
                &[512, H],
                s + 50,
                0.2,
            ));
            t.push(bf(
                &format!("{p}mlp.up_proj.weight"),
                &[512, H],
                s + 51,
                0.2,
            ));
            t.push(bf(
                &format!("{p}mlp.down_proj.weight"),
                &[H, 512],
                s + 52,
                0.2,
            ));
        } else {
            t.push(bf(&format!("{p}moe.gate.weight"), &[NEXP, H], s + 60, 0.2));
            t.push(f32t(
                &format!("{p}moe.gate.e_score_correction_bias"),
                &[NEXP],
                s + 61,
            ));
            for (i, n) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                let (rows, cols) = if *n == "down_proj" {
                    (H, 1088)
                } else {
                    (1088, H)
                };
                t.push(bf(
                    &format!("{p}moe.shared_experts.{n}.weight"),
                    &[rows, cols],
                    s + 70 + i as u32,
                    0.2,
                ));
                for e in 0..NEXP {
                    t.extend(fp8_pair(
                        &format!("{p}moe.experts.{e}.{n}"),
                        rows,
                        cols,
                        s + 80 + (e * 3 + i) as u32,
                    ));
                }
            }
        }
    }
    t
}

fn write_checkpoint(dir: &Path, mut tensors: Vec<T>, cfg: &serde_json::Value) {
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    let mut shards: [Vec<&T>; 2] = [vec![], vec![]];
    for (i, t) in tensors.iter().enumerate() {
        shards[t.shard.unwrap_or(i % 2)].push(t);
    }
    for (si, shard) in shards.iter().enumerate() {
        let views: Vec<(String, TensorView)> = shard
            .iter()
            .map(|t| {
                (
                    t.name.clone(),
                    TensorView::new(t.dtype, t.shape.clone(), &t.data).unwrap(),
                )
            })
            .collect();
        serialize_to_file(
            views,
            None::<HashMap<String, String>>,
            &dir.join(format!("model-0000{}-of-00002.safetensors", si + 1)),
        )
        .unwrap();
    }
    std::fs::write(dir.join("config.json"), serde_json::to_string(cfg).unwrap()).unwrap();
    std::fs::write(dir.join("tokenizer.json"), "{\"model\":{}}").unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), "{\"bos_token\":\"<s>\"}").unwrap();
    std::fs::write(dir.join("chat_template.jinja"), "{{ messages }}").unwrap();
    std::fs::write(dir.join("generation_config.json"), "{\"top_k\":20}").unwrap();
    std::fs::write(dir.join("model.safetensors.index.json"), "{}").unwrap();
}

fn make_source(mutate: impl FnOnce(&mut Vec<T>)) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let mut t = synthetic_tensors();
    mutate(&mut t);
    write_checkpoint(dir.path(), t, &test_config());
    dir
}

fn validate_err(mutate: impl FnOnce(&mut Vec<T>)) -> String {
    let dir = make_source(mutate);
    let src = SourceIndex::open(dir.path());
    match src {
        Err(e) => e,
        Ok(src) => src.validate(&test_config()).unwrap_err(),
    }
}

fn convert_synthetic(dir: &Path, out: &Path) -> RaonSummary {
    run_cli(dir, out).unwrap()
}

// ─── class map ──────────────────────────────────────────────────────────────

#[test]
fn class_map_matches_plan_table() {
    use Storage::*;
    let cases: &[(&str, Storage)] = &[
        ("model.embed_tokens.weight", Q8F16),
        ("lm_head.weight", Q8F16),
        ("model.norm.weight", Lossless),
        ("model.layers.3.moe.experts.127.gate_proj.weight", Qt44),
        ("model.layers.3.moe.experts.0.up_proj.weight", Qt44),
        ("model.layers.3.moe.experts.5.down_proj.weight", Qt44Padded),
        ("model.layers.3.moe.shared_experts.gate_proj.weight", Qt44),
        ("model.layers.3.moe.shared_experts.up_proj.weight", Qt44),
        (
            "model.layers.3.moe.shared_experts.down_proj.weight",
            Qt44Padded,
        ),
        ("model.layers.0.mlp.gate_proj.weight", Qt44),
        ("model.layers.0.mlp.up_proj.weight", Qt44),
        ("model.layers.0.mlp.down_proj.weight", Qt44),
        ("model.layers.3.moe.gate.weight", RouterBf16),
        ("model.layers.3.moe.gate.e_score_correction_bias", F32),
        ("model.layers.1.self_attn.q_proj.weight", Q8F16),
        ("model.layers.1.self_attn.k_proj.weight", Q8F16),
        ("model.layers.1.self_attn.v_proj.weight", Q8F16),
        ("model.layers.1.self_attn.g_proj.weight", Q8F16),
        ("model.layers.1.self_attn.o_proj.weight", Q8F16),
        ("model.layers.3.self_attn.q_a_proj.weight", Q8F16),
        ("model.layers.3.self_attn.kv_a_proj_with_mqa.weight", Q8F16),
        ("model.layers.3.self_attn.kv_b_proj.weight", Q8F16),
        ("model.layers.3.self_attn.q_b_proj.weight", Q8F16),
        ("model.layers.1.self_attn.f_a_proj.weight", Q8F16),
        ("model.layers.1.self_attn.f_b_proj.weight", Q8F16),
        ("model.layers.1.self_attn.b_proj.weight", Q8F16),
        ("model.layers.1.self_attn.rda_w_proj.weight", Q8F16),
        ("model.layers.1.self_attn.A_log", F32),
        ("model.layers.1.self_attn.dt_bias", F32),
        ("model.layers.1.self_attn.q_conv1d.weight", Lossless),
        ("model.layers.1.self_attn.o_norm.weight", Lossless),
        ("model.layers.3.self_attn.q_a_layernorm.weight", Lossless),
        ("model.layers.3.self_attn.kv_a_layernorm.weight", Lossless),
        ("model.layers.0.input_layernorm.weight", Lossless),
        ("model.layers.0.post_attention_layernorm.weight", Lossless),
        ("model.layers.2.self_attn_raon_mhc.phi_pre", Lossless),
        ("model.layers.2.mlp_raon_mhc.res.phi_z", Lossless),
        ("model.layers.2.mlp_raon_mhc.alpha_post", Lossless),
        ("model.layers.2.mlp_raon_mhc.b_pre", Lossless),
        ("model.layers.2.mlp_raon_mhc.res.eta", F32),
    ];
    for (name, want) in cases {
        assert_eq!(classify(name).unwrap(), *want, "{name}");
    }
    for bad in [
        "model.layers.1.self_attn.mystery.weight",
        "model.layers.x.mlp.up_proj.weight",
        "visual.patch.weight",
        "model.layers.1.mlp_raon_mhc.res.zzz",
    ] {
        assert!(classify(bad).is_err(), "{bad}");
    }
    assert_eq!(physical_k(Storage::Qt44Padded, 1088), 1280);
    assert_eq!(physical_k(Storage::Qt44, 2560), 2560);
    assert_eq!(awq_sidecar_name("a.b.weight"), "a.b.awq_scale.weight");
}

/// The official 25-layer / 128-expert tensor set (names only), from the pinned
/// checkpoint's own index when it is available locally.
#[test]
fn real_checkpoint_index_matches_expected_names_and_class_counts() {
    let parent = PathBuf::from("/home/kaden/qcal/parents/raon-1-mini-preview");
    let (Some(cfg), Some(index)) = (
        read_json(&parent.join("config.json")).ok().flatten(),
        read_json(&parent.join("model.safetensors.index.json"))
            .ok()
            .flatten(),
    ) else {
        eprintln!("skip: official checkpoint not present");
        return;
    };
    let names: Vec<String> = index["weight_map"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    let weights: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|n| !n.ends_with(".weight_scale_inv"))
        .collect();
    let present: BTreeSet<&str> = weights.iter().copied().collect();
    let want = expected_names(&cfg, &present).unwrap();
    assert_eq!(want.len(), present.len());
    assert!(want.iter().all(|n| present.contains(n.as_str())));
    let mut by: BTreeMap<String, usize> = BTreeMap::new();
    for n in &weights {
        *by.entry(format!("{:?}", classify(n).unwrap())).or_default() += 1;
    }
    assert_eq!(by["Qt44Padded"], 24 * 129);
    assert_eq!(
        names
            .iter()
            .filter(|n| n.ends_with(".weight_scale_inv"))
            .count(),
        24 * 128 * 3
    );
}

// ─── MXFP8 ──────────────────────────────────────────────────────────────────

#[test]
fn mxfp8_dequant_matches_reference_formula() {
    // E4M3 0x38 = 1.0, 0xb8 = -1.0, 0x40 = 2.0, 0x01 = 2^-9, 0x7e = 448.
    let mut w = vec![0x38u8; 64];
    w[1] = 0xb8;
    w[2] = 0x40;
    w[3] = 0x01;
    w[4] = 0x7e;
    w[32] = 0x38;
    let scale = [127u8, 130u8]; // 2^0, 2^3
    let out = dequantize_mxfp8_e8m0_strict(&w, &[1, 64], &scale, &[1, 2], 32).unwrap();
    assert_eq!(out[0], 1.0);
    assert_eq!(out[1], -1.0);
    assert_eq!(out[2], 2.0);
    assert_eq!(out[3], 2f32.powi(-9));
    assert_eq!(out[4], 448.0);
    assert_eq!(out[32], 8.0);
    assert_eq!(out[63], 8.0);
    // Scale byte 0 = 2^-127 (f32 subnormal), byte 1 = 2^-126.
    let out = dequantize_mxfp8_e8m0_strict(&w[..32], &[1, 32], &[0], &[1, 1], 32).unwrap();
    assert_eq!(out[0], 2f32.powi(-127));
    let out = dequantize_mxfp8_e8m0_strict(&w[..32], &[1, 32], &[1], &[1, 1], 32).unwrap();
    assert_eq!(out[0], f32::MIN_POSITIVE);
    // Matches the PLAN formula for every E4M3 code and every exponent byte < 255.
    for b in 0u16..=255 {
        let b = b as u8;
        if b & 0x7f == 0x7f {
            continue;
        }
        for u in [1u8, 100, 127, 140, 200] {
            let got =
                dequantize_mxfp8_e8m0_strict(&vec![b; 32], &[1, 32], &[u], &[1, 1], 32).unwrap()[0];
            assert_eq!(
                got,
                crate::dequant::e4m3_to_f32(b) * 2f32.powi(u as i32 - 127),
                "{b:#x}/{u}"
            );
        }
    }
}

#[test]
fn mxfp8_dequant_refuses_invalid_codes_and_shapes() {
    let w = vec![0x38u8; 32];
    assert!(
        dequantize_mxfp8_e8m0_strict(&w, &[1, 32], &[255], &[1, 1], 32).is_err(),
        "E8M0 255"
    );
    let mut nan = w.clone();
    nan[7] = 0x7f;
    assert!(
        dequantize_mxfp8_e8m0_strict(&nan, &[1, 32], &[127], &[1, 1], 32).is_err(),
        "E4M3 NaN"
    );
    nan[7] = 0xff;
    assert!(dequantize_mxfp8_e8m0_strict(&nan, &[1, 32], &[127], &[1, 1], 32).is_err());
    // 448 * 2^127 overflows f32.
    assert!(
        dequantize_mxfp8_e8m0_strict(&vec![0x7e; 32], &[1, 32], &[254], &[1, 1], 32).is_err(),
        "overflow"
    );
    // Block-128 style or transposed scale grids are not [1,32].
    assert!(
        dequantize_mxfp8_e8m0_strict(&vec![0x38; 128], &[1, 128], &[127], &[1, 1], 32).is_err()
    );
    assert!(dequantize_mxfp8_e8m0_strict(&w, &[1, 32], &[127, 127], &[2, 1], 32).is_err());
}

#[test]
fn main_pipeline_dequant_reads_u8_scale_as_raw_e8m0() {
    // `tensor_to_f32_with_optional_fp8_scale` must take the U8 `weight_scale_inv`
    // path as E8M0 exponents (not an F32 multiplier).
    let dir = make_source(|_| {});
    let st: Vec<SafetensorsFile> = {
        let mut p: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        p.sort();
        p.iter()
            .map(|p| SafetensorsFile::open(p).unwrap())
            .collect()
    };
    let wname = "model.layers.1.moe.experts.0.gate_proj.weight";
    let sname = "model.layers.1.moe.experts.0.gate_proj.weight_scale_inv";
    let (wf, (meta, bytes)) = st
        .iter()
        .enumerate()
        .find_map(|(i, f)| f.tensor_data(wname).map(|x| (i, x)))
        .unwrap();
    let _ = wf;
    let sf = st
        .iter()
        .position(|f| f.tensor_data(sname).is_some())
        .unwrap();
    let mut map = HashMap::new();
    map.insert(wname.to_string(), (sf, sname.to_string()));
    let got = crate::dequant::tensor_to_f32_with_optional_fp8_scale(wname, bytes, meta, &map, &st);
    let (sm, sb) = st[sf].tensor_data(sname).unwrap();
    let want = dequantize_mxfp8_e8m0_strict(bytes, &meta.shape, sb, &sm.shape, 32).unwrap();
    assert_eq!(got, want);
}

// ─── source validation ──────────────────────────────────────────────────────

const GATE0: &str = "model.layers.1.moe.experts.0.gate_proj";

#[test]
fn synthetic_source_validates_with_cross_shard_pairing() {
    let dir = make_source(|_| {});
    let src = SourceIndex::open(dir.path()).unwrap();
    // weight and scale are adjacent in sort order → alternate shards.
    let (a, b) = (
        format!("{GATE0}.weight"),
        format!("{GATE0}.weight_scale_inv"),
    );
    assert_ne!(src.loc[&a], src.loc[&b]);
    let jobs = src.validate(&test_config()).unwrap();
    assert!(jobs.iter().all(|(n, _)| !n.ends_with("_scale_inv")));
    assert!(jobs.windows(2).all(|w| w[0].0 < w[1].0), "jobs sorted");
}

#[test]
fn source_validation_refuses_scale_and_tensor_defects() {
    let e = validate_err(|t| t.retain(|x| x.name != format!("{GATE0}.weight_scale_inv")));
    assert!(e.contains("no `.weight_scale_inv`"), "{e}");

    let e = validate_err(|t| {
        t.push(T {
            name: "model.layers.1.mlp.up_proj.weight_scale_inv".into(),
            dtype: Dtype::U8,
            shape: vec![1, 1],
            data: vec![127],
            shard: None,
        })
    });
    assert!(e.contains("orphan"), "{e}");

    // Scale whose weight is not a routed expert is orphaned too.
    let e = validate_err(|t| {
        t.push(T {
            name: "model.layers.1.self_attn.q_proj.weight_scale_inv".into(),
            dtype: Dtype::U8,
            shape: vec![512, 8],
            data: vec![127; 4096],
            shard: None,
        })
    });
    assert!(e.contains("orphan"), "{e}");

    let e = validate_err(|t| {
        let mut d = t
            .iter()
            .find(|x| x.name == "model.norm.weight")
            .unwrap()
            .clone();
        // Original in shard 0, duplicate in shard 1.
        for x in t.iter_mut().filter(|x| x.name == "model.norm.weight") {
            x.shard = Some(0);
        }
        d.shard = Some(1);
        t.push(d);
    });
    assert!(e.contains("appears in both"), "{e}");

    let e = validate_err(|t| {
        let x = t
            .iter_mut()
            .find(|x| x.name == format!("{GATE0}.weight_scale_inv"))
            .unwrap();
        x.dtype = Dtype::F32;
        x.data = vec![0; 1088 * 8 * 4];
    });
    assert!(e.contains("raw E8M0"), "{e}");

    let e = validate_err(|t| {
        let x = t
            .iter_mut()
            .find(|x| x.name == format!("{GATE0}.weight_scale_inv"))
            .unwrap();
        x.shape = vec![1088 / 8, 8 * 8];
    });
    assert!(e.contains("raw E8M0"), "{e}");

    let e = validate_err(|t| {
        let x = t
            .iter_mut()
            .find(|x| x.name == format!("{GATE0}.weight"))
            .unwrap();
        x.dtype = Dtype::U8;
    });
    assert!(e.contains("F8_E4M3"), "{e}");

    let e = validate_err(|t| t.retain(|x| x.name != "model.layers.1.self_attn.dt_bias"));
    assert!(e.contains("missing"), "{e}");

    let e = validate_err(|t| {
        t.push(bf(
            "model.layers.1.self_attn.surprise.weight",
            &[4, 4],
            9,
            1.0,
        ))
    });
    assert!(
        e.contains("unexpected") || e.contains("unrecognised"),
        "{e}"
    );
}

#[test]
fn quantization_config_gates_the_mxfp8_interpretation() {
    let dir = make_source(|_| {});
    let src = SourceIndex::open(dir.path()).unwrap();
    for (mutate, needle) in [
        (
            serde_json::json!({"quant_method": "fp8", "weight_block_size": [1, 32]}),
            "mxfp8",
        ),
        (
            serde_json::json!({"quant_method": "mxfp8", "weight_block_size": [128, 128]}),
            "weight_block_size",
        ),
        (
            serde_json::json!({"quant_method": "mxfp8", "weight_block_size": [1, 32], "dequantize": true}),
            "dequantize",
        ),
    ] {
        let mut cfg = test_config();
        cfg["quantization_config"] = mutate;
        let e = src.validate(&cfg).unwrap_err();
        assert!(e.contains(needle), "{e}");
    }
    let mut cfg = test_config();
    cfg.as_object_mut().unwrap().remove("quantization_config");
    assert!(src
        .validate(&cfg)
        .unwrap_err()
        .contains("quantization_config"));
}

#[test]
fn conversion_refuses_bad_exponent_nan_and_nonfinite_sources() {
    for (what, mutate) in [
        (
            "E8M0 255",
            Box::new(|t: &mut Vec<T>| {
                t.iter_mut()
                    .find(|x| x.name == format!("{GATE0}.weight_scale_inv"))
                    .unwrap()
                    .data[5] = 255;
            }) as Box<dyn FnOnce(&mut Vec<T>)>,
        ),
        (
            "E4M3 NaN",
            Box::new(|t: &mut Vec<T>| {
                t.iter_mut()
                    .find(|x| x.name == format!("{GATE0}.weight"))
                    .unwrap()
                    .data[9] = 0x7f;
            }),
        ),
        (
            "BF16 NaN",
            Box::new(|t: &mut Vec<T>| {
                let x = t.iter_mut().find(|x| x.name == "lm_head.weight").unwrap();
                x.data[0..2].copy_from_slice(&bf16::NAN.to_bits().to_le_bytes());
            }),
        ),
    ] {
        let dir = make_source(mutate);
        let out = dir.path().join("out.hfq");
        let e = run_cli(dir.path(), &out).unwrap_err();
        assert!(!out.exists(), "{what}: no artifact on failure");
        assert!(
            e.contains("E8M0") || e.contains("E4M3") || e.contains("non-finite"),
            "{what}: {e}"
        );
    }
}

// ─── symmetric qt44 + padding ───────────────────────────────────────────────

#[test]
fn symmetric_header_verifier_accepts_producer_output_and_rejects_tampering() {
    let w = vals(7, 4 * 256, 1.0);
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    let mut packed = quantize_mq4g256v2_symmetric(&w, 4, 256, &s1, &s2);
    assert_eq!(verify_symmetric_headers("t", &packed).unwrap(), 8);
    // zero group → (scale 0, zero -0.0 = 0x8000)
    let zero = quantize_mq4g256v2_symmetric(&vec![0.0; 256], 1, 256, &s1, &s2);
    assert_eq!(&zero[0..4], &[0, 0, 0, 0x80]);
    assert_eq!(&zero[4..8], &[0, 0, 0, 0x80]);
    verify_symmetric_headers("zero", &zero).unwrap();
    // +0.0 instead of -0.0 is NOT symmetric under the bitwise contract.
    let mut bad = zero.clone();
    bad[3] = 0;
    assert!(verify_symmetric_headers("pz", &bad).is_err());
    packed[2] ^= 1;
    assert!(verify_symmetric_headers("t", &packed).is_err());
}

fn deq_rotated_row(packed: &[u8], groups: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(groups * 256);
    for g in packed.chunks_exact(MQ4V2_GROUP_BYTES).take(groups) {
        for half in 0..2 {
            let d = f16_to_f32(u16::from_le_bytes([g[4 * half], g[4 * half + 1]]));
            let z = f16_to_f32(u16::from_le_bytes([g[4 * half + 2], g[4 * half + 3]]));
            for i in 0..128 {
                let idx = half * 128 + i;
                let byte = g[8 + idx / 2];
                let code = if idx % 2 == 0 { byte & 15 } else { byte >> 4 };
                out.push(d * code as f32 + z);
            }
        }
    }
    out
}

/// Natural K1088 zero-padded to 1280 before the rotation: the quantized product
/// against the rotated, zero-padded natural activation reproduces the natural
/// K1088 product, and padded natural columns are exactly the zero tail.
#[test]
fn padded_down_multiply_parity_and_basis() {
    let (rows, k, phys) = (6usize, 1088usize, 1280usize);
    let w = vals(11, rows * k, 0.5);
    let x = vals(12, k, 1.0);
    let packed = quantize_qt44("down", &w, rows, k, phys).unwrap();
    assert_eq!(packed.len(), rows * (phys / 256) * MQ4V2_GROUP_BYTES);
    // Rotate the zero-padded activation exactly as the runtime does per 256-group.
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    let mut xp = vec![0.0f32; phys];
    xp[..k].copy_from_slice(&x);
    for g in xp.chunks_exact_mut(256) {
        cpu_fwht_256(g, &s1, &s2);
    }
    let mut err2 = 0.0f64;
    let mut ref2 = 0.0f64;
    for r in 0..rows {
        let row = &packed
            [r * (phys / 256) * MQ4V2_GROUP_BYTES..(r + 1) * (phys / 256) * MQ4V2_GROUP_BYTES];
        let deq = deq_rotated_row(row, phys / 256);
        let yq: f64 = deq.iter().zip(&xp).map(|(a, b)| (*a * *b) as f64).sum();
        let yr: f64 = w[r * k..(r + 1) * k]
            .iter()
            .zip(&x)
            .map(|(a, b)| (*a * *b) as f64)
            .sum();
        err2 += (yq - yr).powi(2);
        ref2 += yr * yr;
    }
    assert!(
        (err2 / ref2).sqrt() < 0.2,
        "padded product off: rel {}",
        (err2 / ref2).sqrt()
    );

    // Row 0 of the padded tensor is the first row zero-padded to 1280 BEFORE
    // the rotation, group by group (never adjacent rows folded into a group).
    let mut first = vec![0.0f32; phys];
    first[..k].copy_from_slice(&w[..k]);
    let ref_row = quantize_mq4g256v2_symmetric(&first, 1, phys, &s1, &s2);
    assert_eq!(&packed[..5 * MQ4V2_GROUP_BYTES], &ref_row[..]);
    // Natural padded tail: the last group's final 192 natural entries are zero
    // by construction, so a row whose natural part is empty packs to the
    // all-zero symmetric header.
    let z = quantize_qt44("z", &vec![0.0; 1088], 1, 1088, 1280).unwrap();
    verify_symmetric_headers("z", &z).unwrap();
    assert!(z
        .chunks_exact(MQ4V2_GROUP_BYTES)
        .all(|g| g[8..].iter().all(|&b| b == 0x88)));
    assert!(quantize_qt44("bad", &vec![0.0; 300], 1, 300, 300).is_err());
}

// ─── end-to-end ─────────────────────────────────────────────────────────────

#[test]
fn end_to_end_synthetic_artifact_matches_class_table_and_is_reproducible() {
    let dir = make_source(|t| {
        // A gamma that F16 cannot hold exactly must fall back to F32.
        let x = t
            .iter_mut()
            .find(|x| x.name == "model.layers.0.input_layernorm.weight")
            .unwrap();
        x.data[0..2].copy_from_slice(&bf16::from_f32(1.0e-9).to_bits().to_le_bytes());
    });
    let out = dir.path().join("raon.hfq");
    let s = convert_synthetic(dir.path(), &out);
    let out2 = dir.path().join("raon2.hfq");
    convert_synthetic(dir.path(), &out2);
    assert_eq!(
        std::fs::read(&out).unwrap(),
        std::fs::read(&out2).unwrap(),
        "byte-reproducible"
    );

    let f = RuntimeHfqFile::open_with_reap_plan(&out, None).unwrap();
    assert_eq!(f.arch_id, 19);
    assert!(f.mq4v2_symmetric());
    let by: HashMap<&str, _> = f.tensors().iter().map(|t| (t.name.as_str(), t)).collect();
    let qt = |n: &str| by[n].quant_type;
    let shape = |n: &str| by[n].shape.clone();
    let p = "model.layers.1.";
    assert_eq!(qt("model.embed_tokens.weight"), 3);
    assert_eq!(qt("lm_head.weight"), 3);
    assert_eq!(qt(&format!("{p}moe.experts.1.gate_proj.weight")), 44);
    assert_eq!(
        shape(&format!("{p}moe.experts.1.gate_proj.weight")),
        vec![1088, 256]
    );
    assert_eq!(qt(&format!("{p}moe.experts.1.down_proj.weight")), 44);
    assert_eq!(
        shape(&format!("{p}moe.experts.1.down_proj.weight")),
        vec![256, 1280]
    );
    assert_eq!(
        shape(&format!("{p}moe.shared_experts.down_proj.weight")),
        vec![256, 1280]
    );
    assert_eq!(
        shape(&format!("{p}moe.shared_experts.gate_proj.weight")),
        vec![1088, 256]
    );
    assert_eq!(shape("model.layers.0.mlp.down_proj.weight"), vec![256, 512]);
    assert_eq!(qt(&format!("{p}self_attn.q_proj.weight")), 3);
    assert_eq!(qt(&format!("{p}self_attn.f_b_proj.weight")), 3);
    assert_eq!(qt(&format!("{p}self_attn.rda_w_proj.weight")), 3);
    assert_eq!(qt("model.layers.2.self_attn.q_b_proj.weight"), 3);
    assert_eq!(qt("model.layers.2.self_attn.kv_b_proj.weight"), 3);
    assert!(
        !by.keys()
            .any(|n| n.contains("self_attn") && n.ends_with(".awq_scale.weight")),
        "no sidecars for Q8 attention tensors"
    );
    assert!(
        !by.contains_key("lm_head.awq_scale.weight"),
        "no sidecar for Q8 lm_head"
    );
    assert_eq!(qt(&format!("{p}moe.gate.weight")), 16);
    assert_eq!(
        by[format!("{p}moe.gate.weight").as_str()].shape,
        vec![NEXP as u32, 256]
    );
    assert_eq!(qt(&format!("{p}self_attn.A_log")), 2);
    assert_eq!(qt(&format!("{p}mlp_raon_mhc.res.eta")), 2);
    assert_eq!(
        shape(&format!("{p}mlp_raon_mhc.res.eta")),
        vec![1],
        "0-d scalars stored as one element"
    );
    assert_eq!(
        qt(&format!("{p}mlp_raon_mhc.alpha_pre")),
        1,
        "exact gamma-like values → F16"
    );
    assert_eq!(
        qt("model.layers.0.input_layernorm.weight"),
        2,
        "non-F16-representable → F32"
    );
    assert_eq!(qt("model.layers.2.input_layernorm.weight"), 1);
    // Unit F16 sidecars of physical width.
    let side = format!("{p}moe.experts.1.down_proj.awq_scale.weight");
    assert_eq!(qt(&side), 1);
    assert_eq!(shape(&side), vec![1280]);
    assert!(f
        .tensor_data(&side)
        .unwrap()
        .1
        .chunks_exact(2)
        .all(|c| c == 0x3c00u16.to_le_bytes()));
    assert!(!by.keys().any(|n| n.ends_with("weight_scale_inv")));
    // Counts: padded = (NEXP + 1 shared) down per MoE layer × 2 layers.
    assert_eq!(s.padded, (NEXP + 1) * 2);
    assert_eq!(s.sidecars, s.qt44);
    assert!(s.halves_verified > 0);

    // Metadata identity.
    let meta: serde_json::Value = serde_json::from_str(&f.metadata_json).unwrap();
    assert_eq!(meta["hipfire_source_revision"], SOURCE_REVISION);
    assert_eq!(meta["hipfire_symmetric_verified"], true);
    assert_eq!(meta["hipfire_raon"]["arch_id"], 19);
    assert_eq!(meta["tokenizer_config"]["chat_template"], "{{ messages }}");
    assert_eq!(meta["config"]["model_type"], "raon1_mini");
    assert_eq!(
        meta["hipfire_raon"]["padded_tensors"][format!("{p}moe.experts.0.down_proj.weight")]
            ["logical_k"],
        1088
    );
    assert_eq!(
        meta["hipfire_raon"]["padded_tensors"][format!("{p}moe.experts.0.down_proj.weight")]
            ["physical_k"],
        1280
    );

    // Dequantized experts equal the PLAN formula on the source bytes.
    let src = SourceIndex::open(dir.path()).unwrap();
    let (got, shp) = src.load_f32(&format!("{GATE0}.weight")).unwrap();
    assert_eq!(shp, vec![1088, 256]);
    let (wm, wb) = src.get(&format!("{GATE0}.weight")).unwrap();
    let (_, sb) = src.get(&format!("{GATE0}.weight_scale_inv")).unwrap();
    assert_eq!(wm.dtype, "F8_E4M3");
    for i in [0usize, 31, 32, 255, 256, 1000, 1088 * 256 - 1] {
        let (r, c) = (i / 256, i % 256);
        let want = crate::dequant::e4m3_to_f32(wb[i]) * 2f32.powi(sb[r * 8 + c / 32] as i32 - 127);
        assert_eq!(got[i], want, "elem {i}");
    }

    // Tampering with any symmetric header is caught by the readback pass.
    let mut bytes = std::fs::read(&out).unwrap();
    let info = by["model.layers.0.mlp.gate_proj.weight"];
    bytes[info.data_offset + 2] ^= 0x01;
    let tampered = dir.path().join("tampered.hfq");
    std::fs::write(&tampered, bytes).unwrap();
    let e = verify_artifact(&tampered, true).unwrap_err();
    // The runtime loader itself rejects a non-symmetric header on open.
    assert!(
        e.contains("model.layers.0.mlp.gate_proj.weight") && e.contains("symmetric"),
        "{e}"
    );
}

// ─── public `hipfire quantize` forwarding reaches this producer ─────────────

/// The argv `hipfire quantize --mq4v2-symmetric --tier xt` forwards (see
/// `hipfire-cli` `quantizer_argv`): same class map, same sidecars, qt44.
#[test]
fn public_cli_recipe_argv_reaches_raon_route_and_selects_qt44() {
    use clap::{CommandFactory, FromArgMatches};
    let dir = make_source(|_| {});
    let out = dir.path().join("public.hfq");
    let parse = |extra: &[&str]| {
        let mut argv = vec![
            "hipfire-quantize",
            "--input",
            dir.path().to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
            "--format",
            "mq4",
        ];
        argv.extend_from_slice(extra);
        let m = crate::cli::QuantizeArgs::command()
            .try_get_matches_from(argv)
            .unwrap();
        let a = crate::cli::QuantizeArgs::from_arg_matches(&m).unwrap();
        (m, a)
    };
    // New public recipe controls: Raon route, every flag reachable.
    let (m, args) = parse(&["--mq4v2-symmetric", "--tier", "xt"]);
    assert_eq!(crate::pipeline::route_for(&args), crate::cli::Route::Raon);
    crate::cli::reject_unreachable(&m, crate::cli::Route::Raon).unwrap();
    let s = run_recipe(
        dir.path(),
        &out,
        &args.format,
        args.mq4v2_symmetric,
        args.tier.as_deref(),
    )
    .unwrap();
    assert!(s.qt44 > 0 && s.sidecars == s.qt44);
    let f = RuntimeHfqFile::open_with_reap_plan(&out, None).unwrap();
    assert_eq!(f.arch_id, 19);
    assert!(f.tensors().iter().any(|t| t.quant_type == 44));
    assert!(f
        .tensors()
        .iter()
        .filter(|t| t.quant_type == 44)
        .all(|t| by_sidecar(&f, &t.name)));
    // The old default-only forwarding (`--format mq4` alone) is refused, not
    // silently turned into a different artifact.
    let (_, plain) = parse(&[]);
    assert_eq!(crate::pipeline::route_for(&plain), crate::cli::Route::Raon);
    let e = run_recipe(
        dir.path(),
        &dir.path().join("plain.hfq"),
        &plain.format,
        plain.mq4v2_symmetric,
        plain.tier.as_deref(),
    )
    .unwrap_err();
    assert!(e.contains("--mq4v2-symmetric --tier xt"), "{e}");
    assert!(!dir.path().join("plain.hfq").exists());
}

fn by_sidecar(f: &RuntimeHfqFile, weight: &str) -> bool {
    f.tensors()
        .iter()
        .any(|t| t.name == awq_sidecar_name(weight))
}

// ─── final-code import (same names/physical dims) ───────────────────────────

fn unpack_group_codes(group: &[u8]) -> [u8; 256] {
    let mut c = [0u8; 256];
    for i in 0..128 {
        c[2 * i] = group[8 + i] & 15;
        c[2 * i + 1] = group[8 + i] >> 4;
    }
    c
}

#[test]
fn final_code_import_replaces_scales_and_codes_without_changing_geometry() {
    let dir = make_source(|_| {});
    let control = dir.path().join("control.hfq");
    convert_synthetic(dir.path(), &control);
    let sha = crate::calibration::sha256_file_hex(&control).unwrap();
    let f = RuntimeHfqFile::open_with_reap_plan(&control, None).unwrap();

    // Padded routed down [256,1280] (physical K) plus gate/up siblings that
    // share one producer scale group.
    let targets = [
        "model.layers.1.moe.experts.0.down_proj.weight",
        "model.layers.1.moe.experts.0.gate_proj.weight",
        "model.layers.1.moe.experts.0.up_proj.weight",
    ];
    let records_dir = dir.path().join("records");
    std::fs::create_dir(&records_dir).unwrap();
    let mut expected: Vec<(String, Vec<u8>, Vec<u8>)> = vec![];
    for (ri, name) in targets.iter().enumerate() {
        let (info, data) = f.tensor_data(name).unwrap();
        let (m, k) = (info.shape[0] as usize, info.shape[1] as usize);
        if ri == 0 {
            assert_eq!((m, k), (256, 1280));
        }
        let mut grid = Vec::new();
        let mut codes = Vec::new();
        for g in data.chunks_exact(MQ4V2_GROUP_BYTES) {
            grid.extend_from_slice(&g[..8]);
            codes.extend_from_slice(&unpack_group_codes(g));
        }
        codes[3] = (codes[3] + 5) % 16; // one trained code differs
        let s: Vec<u8> = (0..k)
            .flat_map(|i| f32_to_f16(0.5 + (i % 4) as f32 * 0.25).to_le_bytes())
            .collect();
        let packed = crate::quant_fwht::pack_mq4g256v2_from_f16_grid(m, k, &grid, &codes).unwrap();
        let tensors = [
            ("S_f16", TensorView::new(Dtype::F16, vec![k], &s).unwrap()),
            (
                "d_z_f16",
                TensorView::new(Dtype::F16, vec![m, k / 256, 2, 2], &grid).unwrap(),
            ),
            (
                "codes_u8",
                TensorView::new(Dtype::U8, vec![m, k], &codes).unwrap(),
            ),
        ];
        let metadata = HashMap::from([
            ("name".to_string(), name.to_string()),
            ("M".into(), m.to_string()),
            ("K".into(), k.to_string()),
            ("qt".into(), "44".to_string()),
            ("source_sha".into(), sha.clone()),
        ]);
        serialize_to_file(
            tensors,
            Some(metadata),
            &records_dir.join(format!("r{ri}.safetensors")),
        )
        .unwrap();
        expected.push((name.to_string(), packed, s));
    }

    let out = dir.path().join("imported.hfq");
    let (count, src_sha) =
        crate::pipeline::export_mq4v2_final_codes(&control, &records_dir, &out).unwrap();
    assert_eq!(count, 3);
    assert_eq!(src_sha, sha);
    let g = RuntimeHfqFile::open_with_reap_plan(&out, None).unwrap();
    // Geometry (names, shapes, qt, sizes, offsets, metadata) identical.
    assert_eq!(f.metadata_json, g.metadata_json);
    assert_eq!(f.tensors().len(), g.tensors().len());
    for (a, b) in f.tensors().iter().zip(g.tensors()) {
        assert_eq!(
            (
                &a.name,
                a.quant_type,
                &a.shape,
                a.group_size,
                a.data_offset,
                a.data_size
            ),
            (
                &b.name,
                b.quant_type,
                &b.shape,
                b.group_size,
                b.data_offset,
                b.data_size
            )
        );
    }
    for (name, packed, s) in &expected {
        assert_eq!(
            g.tensor_data(name).unwrap().1,
            packed.as_slice(),
            "{name} codes"
        );
        assert_ne!(f.tensor_data(name).unwrap().1, packed.as_slice());
        assert_eq!(
            g.tensor_data(&awq_sidecar_name(name)).unwrap().1,
            s.as_slice(),
            "{name} sidecar"
        );
    }
    // Untouched tensors stay byte-identical.
    let other = "model.layers.1.moe.experts.1.down_proj.weight";
    assert_eq!(
        f.tensor_data(other).unwrap().1,
        g.tensor_data(other).unwrap().1
    );
    // Imported scales are not unit anymore but all qt44 headers remain symmetric.
    assert!(verify_artifact(&out, true).is_err());
    verify_artifact(&out, false).unwrap();
}

/// `cargo test -p hipfire-quantize raon_verify_artifact -- --ignored` with
/// `RAON_VERIFY_ARTIFACT=<path.hfq>` reads a real artifact back and prints counts.
#[test]
#[ignore]
fn raon_verify_artifact() {
    let path = std::env::var("RAON_VERIFY_ARTIFACT").expect("set RAON_VERIFY_ARTIFACT");
    let s = verify_artifact(
        Path::new(&path),
        std::env::var("RAON_VERIFY_UNIT").map_or(true, |v| v != "0"),
    )
    .unwrap();
    eprintln!("RAON VERIFY OK: {s:?}");
    assert_eq!(s.padded, 24 * 129);
}

// ─── always-active FFN storage: class map, swap route, verify route ─────────

const SHARED_TARGETS: [&str; 9] = [
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.1.moe.shared_experts.gate_proj.weight",
    "model.layers.1.moe.shared_experts.up_proj.weight",
    "model.layers.1.moe.shared_experts.down_proj.weight",
    "model.layers.2.moe.shared_experts.gate_proj.weight",
    "model.layers.2.moe.shared_experts.up_proj.weight",
    "model.layers.2.moe.shared_experts.down_proj.weight",
];

fn shipped_artifact(dir: &Path) -> PathBuf {
    let out = dir.join("shipped.hfq");
    convert_synthetic(dir, &out);
    out
}

/// Rewrite `input` into `out` tensor by tensor (fixtures only: everything is held in
/// RAM). `tensor` returns the `(qt, payload)` to keep, or `None` to drop the tensor.
fn rewrite_hfq(
    input: &Path,
    out: &Path,
    arch: u32,
    mut tensor: impl FnMut(&hipfire_runtime::hfq::HfqTensorInfo, Vec<u8>) -> Option<(u8, Vec<u8>)>,
    meta: impl FnOnce(&mut serde_json::Value),
) {
    let f = RuntimeHfqFile::open_with_reap_plan(input, None).unwrap();
    let mut m: serde_json::Value = serde_json::from_str(&f.metadata_json).unwrap();
    meta(&mut m);
    let mut entries = Vec::new();
    let mut data: Vec<Vec<u8>> = Vec::new();
    for t in f.tensors() {
        let bytes = f.tensor_data(&t.name).unwrap().1.to_vec();
        if let Some((qt, bytes)) = tensor(t, bytes) {
            entries.push(HfqIndexEntry {
                name: t.name.clone(),
                quant_type: qt,
                shape: t.shape.clone(),
                group_size: t.group_size,
                data_len: bytes.len() as u64,
            });
            data.push(bytes);
        }
    }
    write_hfq_streamed(
        out,
        arch,
        &serde_json::to_string(&m).unwrap(),
        &entries,
        |i, w| w.write_all(&data[i]),
    )
    .unwrap();
}

fn deq_q8(bytes: &[u8], rows: usize, phys: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * phys);
    for blk in bytes.chunks_exact(Q8F16_BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        out.extend(blk[2..].iter().map(|&q| q as i8 as f32 * d));
    }
    assert_eq!(out.len(), rows * phys);
    out
}

#[test]
fn shared_ffn_class_map_is_pure_in_name_and_choice() {
    use Storage::*;
    let gate = "model.layers.3.moe.shared_experts.gate_proj.weight";
    let up = "model.layers.3.moe.shared_experts.up_proj.weight";
    let down = "model.layers.3.moe.shared_experts.down_proj.weight";
    let dense = [
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ];
    for (choice, plain, padded) in [
        (SharedFfn::Qt44, Qt44, Qt44Padded),
        (SharedFfn::Q8F16, Q8F16, Q8F16Padded),
        (SharedFfn::Bf16, Bf16, Bf16Padded),
    ] {
        for n in [gate, up, dense[0], dense[1], dense[2]] {
            assert_eq!(classify_with(n, Q8F16, choice).unwrap(), plain, "{n}");
        }
        assert_eq!(classify_with(down, Q8F16, choice).unwrap(), padded);
        // Classes the choice must not touch.
        for (n, want) in [
            ("model.layers.3.moe.experts.5.down_proj.weight", Qt44Padded),
            ("model.layers.3.moe.experts.5.gate_proj.weight", Qt44),
            ("model.layers.3.moe.experts.0.up_proj.weight", Qt44),
            ("model.layers.1.self_attn.q_proj.weight", Q8F16),
            ("model.embed_tokens.weight", Q8F16),
            ("model.norm.weight", Lossless),
            ("model.layers.3.moe.gate.weight", RouterBf16),
        ] {
            assert_eq!(classify_with(n, Q8F16, choice).unwrap(), want, "{n}");
        }
        // The head follows its own choice only.
        assert_eq!(classify_with("lm_head.weight", Qt44, choice).unwrap(), Qt44);
        assert_eq!(
            classify_with("lm_head.weight", Q8F16, choice).unwrap(),
            Q8F16
        );
        assert!(classify_with("model.layers.1.self_attn.mystery.weight", Q8F16, choice).is_err());
    }
    assert_eq!(physical_k(Q8F16Padded, 1088), 1280);
    assert_eq!(physical_k(Bf16Padded, 1088), 1280);
    assert_eq!(physical_k(Bf16Padded, 1280), 1280);
    assert_eq!(physical_k(Bf16, 12032), 12032);
    assert_eq!(physical_k(Q8F16, 12032), 12032);
    for s in ["qt44", "q8f16", "bf16"] {
        assert_eq!(SharedFfn::parse(s).unwrap().as_str(), s);
    }
    for bad in ["", "BF16", "q8", "qt4", "f16"] {
        assert!(SharedFfn::parse(bad).is_err(), "{bad:?}");
    }
    assert!(is_shared_ffn_weight(gate) && is_shared_ffn_weight(dense[2]));
    assert!(!is_shared_ffn_weight(
        "model.layers.3.moe.experts.0.up_proj.weight"
    ));
    assert_eq!(
        shared_ffn_sidecar_parent("model.layers.3.moe.shared_experts.up_proj.awq_scale.weight")
            .as_deref(),
        Some(up)
    );
    assert_eq!(
        shared_ffn_sidecar_parent("model.layers.3.moe.experts.0.up_proj.awq_scale.weight"),
        None
    );
}

#[test]
fn bf16_bits_round_nearest_even_and_exact_for_bf16_sources() {
    for bits in [0u16, 0x8000, 0x3f80, 0xbf80, 0x0001, 0x7f7f, 0xff7f] {
        assert_eq!(
            f32_to_bf16_bits(f32::from_bits((bits as u32) << 16)),
            bits,
            "{bits:#x}"
        );
    }
    // Ties go to even.
    assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f80_8000)), 0x3f80);
    assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f81_8000)), 0x3f82);
    for v in vals(7, 2000, 3.0) {
        assert_eq!(f32_to_bf16_bits(v), bf16::from_f32(v).to_bits(), "{v}");
    }
}

fn router_name() -> String {
    "model.layers.1.moe.gate.weight".to_string()
}

fn convert_router(dtype: Dtype, data: Vec<u8>) -> JobOut {
    let name = router_name();
    let dir = make_source(|t| {
        let r = t.iter_mut().find(|x| x.name == name).unwrap();
        r.dtype = dtype;
        r.data = data;
    });
    let src = SourceIndex::open(dir.path()).unwrap();
    convert(&src, &name, Storage::RouterBf16).unwrap()
}

#[test]
fn router_bf16_exact_source_is_stored_as_raw_bf16() {
    let name = router_name();
    let src_bytes = bf16_bytes(&vals(91, NEXP * H, 0.2));
    let out = convert_router(Dtype::BF16, src_bytes.clone());
    assert_eq!(out.tensors.len(), 1, "no sidecar");
    let t = &out.tensors[0];
    assert_eq!(t.name, name);
    assert_eq!(t.quant_type as u32, 16);
    assert_eq!(t.shape, vec![NEXP as u32, H as u32], "unpadded [rows,k]");
    assert_eq!(t.group_size, 0);
    assert_eq!(t.data, src_bytes, "byte-exact source bf16 bits");
    assert!(out.padded.is_none());
}

#[test]
fn router_with_inexact_value_falls_back_to_f32_without_rounding() {
    let mut v: Vec<f32> = bf16_bytes(&vals(92, NEXP * H, 0.2))
        .chunks_exact(2)
        .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
        .collect();
    v[7] = 1.000_000_1;
    assert_ne!(
        f32::from_bits((f32_to_bf16_bits(v[7]) as u32) << 16).to_bits(),
        v[7].to_bits(),
        "test value must not be bf16-exact"
    );
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let out = convert_router(Dtype::F32, bytes.clone());
    let t = &out.tensors[0];
    assert_eq!(t.quant_type as u32, 2);
    assert_eq!(t.shape, vec![NEXP as u32, H as u32]);
    assert_eq!(
        t.data, bytes,
        "F32 payload keeps the source value bit-exact"
    );
}

fn check_swap(target: SharedFfn) {
    let dir = make_source(|_| {});
    let shipped = shipped_artifact(dir.path());
    let out = dir.path().join(format!("swap-{}.hfq", target.as_str()));
    let s = swap_shared_ffn(&shipped, dir.path(), &out, target, Scope::All).unwrap();
    assert_eq!(s.replaced, 9);
    assert_eq!(s.dropped_sidecars, 9);
    assert!(s.replaced_bytes_after > 0 && s.replaced_bytes_before > 0);
    assert_eq!(s.output_bytes, std::fs::metadata(&out).unwrap().len());
    assert_eq!(s.verified.shared_ffn, target.as_str());
    assert_eq!(s.verified.sidecars, s.verified.qt44);

    let fa = RuntimeHfqFile::open_with_reap_plan(&shipped, None).unwrap();
    let fb = RuntimeHfqFile::open_with_reap_plan(&out, None).unwrap();
    assert_eq!(fb.arch_id, 19);
    let dropped: BTreeSet<String> = SHARED_TARGETS.iter().map(|n| awq_sidecar_name(n)).collect();
    let want_names: Vec<&str> = fa
        .tensors()
        .iter()
        .map(|t| t.name.as_str())
        .filter(|n| !dropped.contains(*n))
        .collect();
    let got_names: Vec<&str> = fb.tensors().iter().map(|t| t.name.as_str()).collect();
    assert_eq!(got_names, want_names, "order kept, only sidecars removed");
    assert_eq!(s.passthrough, want_names.len() - 9);

    let src = SourceIndex::open(dir.path()).unwrap();
    let targets: BTreeSet<&str> = SHARED_TARGETS.iter().copied().collect();
    for tb in fb.tensors() {
        let (ta, da) = fa.tensor_data(&tb.name).unwrap();
        let (_, db) = fb.tensor_data(&tb.name).unwrap();
        if !targets.contains(tb.name.as_str()) {
            assert_eq!(
                (ta.quant_type, &ta.shape, ta.group_size),
                (tb.quant_type, &tb.shape, tb.group_size),
                "{}",
                tb.name
            );
            assert_eq!(da, db, "{} must be byte-identical", tb.name);
            continue;
        }
        assert_eq!(ta.quant_type, 44, "{}", tb.name);
        assert_eq!(tb.shape, ta.shape, "{}", tb.name);
        let (rows, phys) = (tb.shape[0] as usize, tb.shape[1] as usize);
        let (wm, wb) = src.get(&tb.name).unwrap();
        assert_eq!(wm.dtype, "BF16");
        let k = wm.shape[1];
        let shared_down = tb.name.contains("shared_experts") && tb.name.contains("down_proj");
        assert_eq!(phys, if shared_down { 1280 } else { k }, "{}", tb.name);
        match target {
            SharedFfn::Bf16 => {
                assert_eq!((tb.quant_type, tb.group_size), (16, 0), "{}", tb.name);
                assert_eq!(db.len(), rows * phys * 2);
                for r in 0..rows {
                    assert_eq!(
                        &db[r * phys * 2..r * phys * 2 + 2 * k],
                        &wb[r * k * 2..(r + 1) * k * 2],
                        "{} row {r}: exact source bf16 bits",
                        tb.name
                    );
                    assert!(
                        db[r * phys * 2 + 2 * k..(r + 1) * phys * 2]
                            .iter()
                            .all(|&b| b == 0),
                        "{} row {r}: zero padded columns",
                        tb.name
                    );
                }
            }
            _ => {
                assert_eq!((tb.quant_type, tb.group_size), (3, 32), "{}", tb.name);
                assert_eq!(db.len(), rows * (phys / 32) * 34);
                let deq = deq_q8(db, rows, phys);
                let (sv, _) = src.load_f32(&tb.name).unwrap();
                for r in 0..rows {
                    for c in 0..k {
                        let (want, got) = (sv[r * k + c], deq[r * phys + c]);
                        assert!(
                            (want - got).abs() <= 0.2 / 127.0 + 1e-6,
                            "{} [{r},{c}] {want} vs {got}",
                            tb.name
                        );
                    }
                    assert!(
                        deq[r * phys + k..(r + 1) * phys].iter().all(|&v| v == 0.0),
                        "{} row {r}: zero padded columns",
                        tb.name
                    );
                }
            }
        }
    }

    let meta: serde_json::Value = serde_json::from_str(&fb.metadata_json).unwrap();
    assert_eq!(meta["hipfire_raon"]["shared_ffn"], target.as_str());
    assert_eq!(
        meta["hipfire_raon"]["shared_ffn_swap"]["to"],
        target.as_str()
    );
    assert_eq!(meta["hipfire_raon"]["shared_ffn_swap"]["from"], "qt44");
    assert_eq!(
        meta["hipfire_raon"]["shared_ffn_swap"]["source_revision"],
        SOURCE_REVISION
    );
    let padded = &meta["hipfire_raon"]["padded_tensors"]
        ["model.layers.1.moe.shared_experts.down_proj.weight"];
    assert_eq!(padded["logical_k"], 1088);
    assert_eq!(padded["physical_k"], 1280);
    assert!(meta["hipfire_quant_recipe"]
        .as_str()
        .unwrap()
        .ends_with(&format!("+shared-ffn={}", target.as_str())));
    let census_total: u64 = meta["hipfire_recipe_census"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_u64().unwrap())
        .sum();
    assert_eq!(census_total as usize, fb.tensors().len());
    // The two synthetic MoE layers' routers are BF16 in every artifact; the always-active
    // FFN adds nine BF16 tensors only for the BF16 target.
    assert_eq!(
        meta["hipfire_recipe_census"]["BF16"].as_u64(),
        Some(if target == SharedFfn::Bf16 { 11 } else { 2 })
    );

    // The read-only proof accepts it and counts every class.
    let rep = raon_hfq_diff(&shipped, &out, target, Scope::All).unwrap();
    assert!(rep.ok(), "{:?}\n{}", rep.violations, rep.table());
    assert_eq!(rep.rows[CLASS_SIDECAR].tensors, 9);
    assert_eq!(rep.rows[CLASS_PASS].tensors, s.passthrough);
    assert_eq!(rep.rows[CLASS_GATE_UP].tensors, 4);
    assert_eq!(rep.rows[CLASS_DOWN].tensors, 2);
    assert_eq!(rep.rows[CLASS_DENSE].tensors, 3);
    assert!(rep.table().contains("shared down"));

    // Reproducible.
    let out2 = dir.path().join("swap-again.hfq");
    swap_shared_ffn(&shipped, dir.path(), &out2, target, Scope::All).unwrap();
    assert_eq!(std::fs::read(&out).unwrap(), std::fs::read(&out2).unwrap());
}

#[test]
fn swap_to_q8f16_replaces_only_the_always_active_ffn() {
    check_swap(SharedFfn::Q8F16);
}

#[test]
fn swap_to_bf16_is_source_exact_and_padded() {
    check_swap(SharedFfn::Bf16);
}

#[test]
fn swap_refuses_bad_inputs_and_writes_nothing() {
    let dir = make_source(|_| {});
    let d = dir.path();
    let shipped = shipped_artifact(d);
    let out = d.join("o.hfq");
    let refused = |r: Result<SwapSummary, String>, needle: &str| {
        let e = r.unwrap_err();
        assert!(e.contains(needle), "{e}");
    };
    refused(
        swap_shared_ffn(&shipped, d, &out, SharedFfn::Qt44, Scope::All),
        "q8f16 or bf16",
    );
    refused(
        swap_shared_ffn(&shipped, d, &shipped, SharedFfn::Bf16, Scope::All),
        "must differ",
    );

    // Idempotence: a variant is never swapped again.
    let bf = d.join("bf.hfq");
    swap_shared_ffn(&shipped, d, &bf, SharedFfn::Bf16, Scope::All).unwrap();
    refused(
        swap_shared_ffn(&bf, d, &d.join("again.hfq"), SharedFfn::Bf16, Scope::All),
        "already stores",
    );
    refused(
        swap_shared_ffn(&bf, d, &d.join("q.hfq"), SharedFfn::Q8F16, Scope::All),
        "starts from a qt44",
    );

    // A source that is not the one the artifact was built from.
    let other = tempfile::tempdir().unwrap();
    let mut cfg = test_config();
    cfg["note"] = "different".into();
    write_checkpoint(other.path(), synthetic_tensors(), &cfg);
    refused(
        swap_shared_ffn(&shipped, other.path(), &out, SharedFfn::Q8F16, Scope::All),
        "sha256",
    );
    // An incomplete source.
    let broken =
        make_source(|t| t.retain(|x| x.name != "model.layers.1.moe.shared_experts.up_proj.weight"));
    refused(
        swap_shared_ffn(&shipped, broken.path(), &out, SharedFfn::Q8F16, Scope::All),
        "missing",
    );

    // An artifact without one of the target tensors (and its sidecar).
    let miss = d.join("miss.hfq");
    rewrite_hfq(
        &shipped,
        &miss,
        19,
        |t, b| {
            (!t.name
                .starts_with("model.layers.1.moe.shared_experts.up_proj."))
            .then_some((t.quant_type, b))
        },
        |_| {},
    );
    refused(
        swap_shared_ffn(&miss, d, &out, SharedFfn::Q8F16, Scope::All),
        "missing from the input artifact",
    );
    // A target that is not qt44.
    let notq = d.join("notq.hfq");
    rewrite_hfq(
        &shipped,
        &notq,
        19,
        |t, b| {
            let qt = if t.name == "model.layers.0.mlp.gate_proj.weight" {
                3
            } else {
                t.quant_type
            };
            Some((qt, b))
        },
        |_| {},
    );
    refused(
        swap_shared_ffn(&notq, d, &out, SharedFfn::Q8F16, Scope::All),
        "qt3",
    );
    // Not a Raon artifact.
    let wrong = d.join("wrong-arch.hfq");
    rewrite_hfq(&shipped, &wrong, 18, |t, b| Some((t.quant_type, b)), |_| {});
    refused(
        swap_shared_ffn(&wrong, d, &out, SharedFfn::Q8F16, Scope::All),
        "arch_id",
    );
    assert!(!out.exists(), "a refused swap leaves no output");
    assert!(!d.join("again.hfq").exists() && !d.join("q.hfq").exists());
}

#[test]
fn shared_ffn_verify_flags_every_class_of_violation() {
    let dir = make_source(|_| {});
    let d = dir.path();
    let shipped = shipped_artifact(d);
    let good = d.join("good.hfq");
    swap_shared_ffn(&shipped, d, &good, SharedFfn::Bf16, Scope::All).unwrap();
    let verdict = |b: &Path, t: SharedFfn| raon_hfq_diff(&shipped, b, t, Scope::All).unwrap();
    assert!(verdict(&good, SharedFfn::Bf16).ok());
    assert!(raon_hfq_diff(&shipped, &good, SharedFfn::Qt44, Scope::All).is_err());

    // Wrong target storage for this variant; unchanged artifact; reversed roles.
    assert!(!verdict(&good, SharedFfn::Q8F16).ok());
    let same = verdict(&shipped, SharedFfn::Bf16);
    assert!(!same.ok());
    assert!(same.violations.iter().any(|v| v.contains("survives in B")));
    assert!(!raon_hfq_diff(&good, &shipped, SharedFfn::Bf16, Scope::All)
        .unwrap()
        .ok());

    // One flipped payload byte in a pass-through tensor.
    let off = RuntimeHfqFile::open_with_reap_plan(&good, None)
        .unwrap()
        .find_tensor_info("model.embed_tokens.weight")
        .unwrap()
        .data_offset;
    let mut bytes = std::fs::read(&good).unwrap();
    bytes[off] ^= 1;
    let tampered = d.join("tampered-payload.hfq");
    std::fs::write(&tampered, bytes).unwrap();
    let rep = verdict(&tampered, SharedFfn::Bf16);
    assert!(!rep.ok());
    assert!(
        rep.violations
            .iter()
            .any(|v| v.contains("model.embed_tokens.weight") && v.contains("payload differs")),
        "{:?}",
        rep.violations
    );
    assert_eq!(rep.rows[CLASS_PASS].violations, 1);
    assert!(rep.table().contains("FAIL(1)"));

    // A pass-through tensor missing from the variant.
    let short = d.join("short.hfq");
    rewrite_hfq(
        &good,
        &short,
        19,
        |t, b| (t.name != "model.norm.weight").then_some((t.quant_type, b)),
        |_| {},
    );
    assert!(!verdict(&short, SharedFfn::Bf16).ok());

    // A target payload of the wrong size.
    let sized = d.join("sized.hfq");
    rewrite_hfq(
        &good,
        &sized,
        19,
        |t, mut b| {
            if t.name == "model.layers.0.mlp.down_proj.weight" {
                b.truncate(b.len() - 2);
            }
            Some((t.quant_type, b))
        },
        |_| {},
    );
    let rep = verdict(&sized, SharedFfn::Bf16);
    assert!(rep
        .violations
        .iter()
        .any(|v| v.contains("model.layers.0.mlp.down_proj.weight") && v.contains("size")));

    // Metadata outside the updated keys changed / updated keys wrong.
    let meta_a = d.join("meta-a.hfq");
    rewrite_hfq(
        &good,
        &meta_a,
        19,
        |t, b| Some((t.quant_type, b)),
        |m| m["hipfire_lm_head"] = "qt44".into(),
    );
    let rep = verdict(&meta_a, SharedFfn::Bf16);
    assert!(rep
        .violations
        .iter()
        .any(|v| v.contains("metadata differs") && v.contains("hipfire_lm_head")));
    let meta_b = d.join("meta-b.hfq");
    rewrite_hfq(
        &good,
        &meta_b,
        19,
        |t, b| Some((t.quant_type, b)),
        |m| m["hipfire_raon"]["shared_ffn"] = "q8f16".into(),
    );
    assert!(!verdict(&meta_b, SharedFfn::Bf16).ok());
    let meta_c = d.join("meta-c.hfq");
    rewrite_hfq(
        &good,
        &meta_c,
        19,
        |t, b| Some((t.quant_type, b)),
        |m| m["hipfire_recipe_census"]["BF16"] = 0.into(),
    );
    assert!(!verdict(&meta_c, SharedFfn::Bf16).ok());
}

#[test]
fn verify_artifact_checks_swapped_classes_against_the_recorded_storage() {
    let dir = make_source(|_| {});
    let d = dir.path();
    let shipped = shipped_artifact(d);
    for target in [SharedFfn::Q8F16, SharedFfn::Bf16] {
        let out = d.join(format!("v-{}.hfq", target.as_str()));
        swap_shared_ffn(&shipped, d, &out, target, Scope::All).unwrap();
        let s = verify_artifact(&out, false).unwrap();
        assert_eq!(s.shared_ffn, target.as_str());
        assert_eq!(s.padded, (NEXP + 1) * 2);
        // The same tensors under metadata that still claims qt44 are a class violation.
        let lying = d.join(format!("lie-{}.hfq", target.as_str()));
        rewrite_hfq(
            &out,
            &lying,
            19,
            |t, b| Some((t.quant_type, b)),
            |m| m["hipfire_raon"]["shared_ffn"] = "qt44".into(),
        );
        let e = verify_artifact(&lying, false).unwrap_err();
        assert!(e.contains("violates class"), "{e}");
        // Unknown storage names are refused.
        let bad = d.join(format!("bad-{}.hfq", target.as_str()));
        rewrite_hfq(
            &out,
            &bad,
            19,
            |t, b| Some((t.quant_type, b)),
            |m| m["hipfire_raon"]["shared_ffn"] = "int4".into(),
        );
        assert!(verify_artifact(&bad, false)
            .unwrap_err()
            .contains("shared_ffn"));
    }
}

#[test]
fn shared_ffn_flags_select_their_route_and_conflict_with_other_routes() {
    use clap::{CommandFactory, FromArgMatches};
    let parse = |argv: &[&str]| {
        crate::cli::QuantizeArgs::command()
            .try_get_matches_from(std::iter::once("hipfire-quantize").chain(argv.iter().copied()))
    };
    let swap = [
        "--raon-shared-ffn",
        "q8f16",
        "--raon-source",
        "d",
        "--input",
        "i",
        "--output",
        "o",
    ];
    let m = parse(&swap).unwrap();
    let args = crate::cli::QuantizeArgs::from_arg_matches(&m).unwrap();
    assert_eq!(
        crate::pipeline::route_for(&args),
        crate::cli::Route::RaonSharedFfn
    );
    crate::cli::reject_unreachable(&m, crate::cli::Route::RaonSharedFfn).unwrap();
    let verify = [
        "--raon-shared-ffn-verify",
        "bf16",
        "--input",
        "i",
        "--output",
        "o",
    ];
    let m = parse(&verify).unwrap();
    let args = crate::cli::QuantizeArgs::from_arg_matches(&m).unwrap();
    assert_eq!(
        crate::pipeline::route_for(&args),
        crate::cli::Route::RaonSharedFfn
    );
    crate::cli::reject_unreachable(&m, crate::cli::Route::RaonSharedFfn).unwrap();
    // Flags the route never reads are refused.
    let mut with_awq = swap.to_vec();
    with_awq.push("--awq");
    let m = parse(&with_awq).unwrap();
    let e = crate::cli::reject_unreachable(&m, crate::cli::Route::RaonSharedFfn).unwrap_err();
    assert!(e.starts_with("--awq has no effect"), "{e}");
    // The swap flags mean nothing to any other route.
    let m = parse(&swap).unwrap();
    for route in [crate::cli::Route::Raon, crate::cli::Route::Main] {
        let e = crate::cli::reject_unreachable(&m, route).unwrap_err();
        assert!(e.contains("has no effect"), "{route:?}: {e}");
    }
    // Parse-time refusals.
    assert!(parse(&[
        "--raon-shared-ffn",
        "q8f16",
        "--input",
        "i",
        "--output",
        "o"
    ])
    .is_err());
    assert!(parse(&["--raon-source", "d", "--input", "i", "--output", "o"]).is_err());
    assert!(parse(&[
        "--raon-shared-ffn",
        "qt44",
        "--raon-source",
        "d",
        "--input",
        "i",
        "--output",
        "o"
    ])
    .is_err());
    let mut both = swap.to_vec();
    both.extend(["--raon-shared-ffn-verify", "bf16"]);
    assert!(parse(&both).is_err());
    let mut codes = swap.to_vec();
    codes.extend(["--mq4v2-final-codes", "r"]);
    assert!(parse(&codes).is_err());
}

#[test]
fn partial_scope_swaps_only_its_class_and_is_described_and_verified() {
    use Storage::*;
    assert!(Scope::Shared.covers("model.layers.1.moe.shared_experts.up_proj.weight"));
    assert!(!Scope::Shared.covers("model.layers.0.mlp.up_proj.weight"));
    assert!(Scope::Dense.covers("model.layers.0.mlp.up_proj.weight"));
    assert!(!Scope::Dense.covers("model.layers.1.moe.shared_experts.up_proj.weight"));
    assert!(Scope::All.covers("model.layers.0.mlp.up_proj.weight"));
    assert!(!Scope::All.covers("model.layers.1.moe.experts.0.up_proj.weight"));
    assert_eq!(
        classify_scoped(
            "model.layers.0.mlp.up_proj.weight",
            Q8F16,
            SharedFfn::Bf16,
            Scope::Shared
        )
        .unwrap(),
        Qt44
    );
    assert_eq!(
        classify_scoped(
            "model.layers.1.moe.shared_experts.down_proj.weight",
            Q8F16,
            SharedFfn::Bf16,
            Scope::Shared
        )
        .unwrap(),
        Bf16Padded
    );
    assert_eq!(storage_tag(SharedFfn::Q8F16, Scope::All), "q8f16");
    assert_eq!(storage_tag(SharedFfn::Q8F16, Scope::Dense), "q8f16:dense");
    assert!(Scope::parse("both").is_err());
    assert_eq!(Scope::parse("shared-gateup").unwrap(), Scope::SharedGateUp);
    assert!(Scope::SharedGateUp.covers("model.layers.1.moe.shared_experts.gate_proj.weight"));
    assert!(Scope::SharedGateUp.covers("model.layers.1.moe.shared_experts.up_proj.weight"));
    assert!(!Scope::SharedGateUp.covers("model.layers.1.moe.shared_experts.down_proj.weight"));
    assert!(!Scope::SharedGateUp.covers("model.layers.0.mlp.gate_proj.weight"));
    assert!(!Scope::Shared.covers("model.layers.0.mlp.down_proj.weight"));

    let dir = make_source(|_| {});
    let d = dir.path();
    let shipped = shipped_artifact(d);
    for (scope, target, replaced, tag) in [
        (Scope::Shared, SharedFfn::Q8F16, 6usize, "q8f16:shared"),
        (Scope::Dense, SharedFfn::Bf16, 3usize, "bf16:dense"),
        (
            Scope::SharedGateUp,
            SharedFfn::Q8F16,
            4usize,
            "q8f16:shared-gateup",
        ),
    ] {
        let out = d.join(format!("scope-{}.hfq", scope.as_str()));
        let s = swap_shared_ffn(&shipped, d, &out, target, scope).unwrap();
        assert_eq!((s.replaced, s.dropped_sidecars), (replaced, replaced));
        assert_eq!(s.verified.shared_ffn, tag);
        let fa = RuntimeHfqFile::open_with_reap_plan(&shipped, None).unwrap();
        let fb = RuntimeHfqFile::open_with_reap_plan(&out, None).unwrap();
        let meta: serde_json::Value = serde_json::from_str(&fb.metadata_json).unwrap();
        assert_eq!(meta["hipfire_raon"]["shared_ffn"], tag);
        assert_eq!(
            meta["hipfire_raon"]["shared_ffn_swap"]["scope"],
            scope.as_str()
        );
        assert!(meta["hipfire_quant_recipe"]
            .as_str()
            .unwrap()
            .ends_with(&format!("+shared-ffn={tag}")));
        for t in SHARED_TARGETS {
            let (ia, da) = fa.tensor_data(t).unwrap();
            let (ib, db) = fb.tensor_data(t).unwrap();
            if scope.covers(t) {
                assert_eq!(ib.quant_type, swapped_qt(target), "{t}");
                assert!(fb.find_tensor_info(&awq_sidecar_name(t)).is_none(), "{t}");
            } else {
                assert_eq!((ia.quant_type, ib.quant_type), (44, 44), "{t}");
                assert_eq!(da, db, "{t} stays byte-identical");
                assert!(fb.find_tensor_info(&awq_sidecar_name(t)).is_some(), "{t}");
            }
        }
        let rep = raon_hfq_diff(&shipped, &out, target, scope).unwrap();
        assert!(rep.ok(), "{:?}\n{}", rep.violations, rep.table());
        assert_eq!(rep.rows[CLASS_SIDECAR].tensors, replaced);
        // Verify expects the same scope: the other scopes must fail.
        for other in [Scope::All, Scope::Shared, Scope::SharedGateUp, Scope::Dense] {
            if other != scope {
                assert!(
                    !raon_hfq_diff(&shipped, &out, target, other).unwrap().ok(),
                    "{scope:?} variant must not verify as {other:?}"
                );
            }
        }
        // A scoped variant is not swapped again.
        let e = swap_shared_ffn(&out, d, &d.join("x.hfq"), target, Scope::All).unwrap_err();
        assert!(e.contains("starts from a qt44"), "{e}");
    }
}
