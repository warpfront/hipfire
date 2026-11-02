// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Capture reader (contract: local://a4ec-contract.md) and a synthetic capture generator.
use crate::{quant, rot};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

pub const TAGS: [(&str, usize); 4] = [("attn", 5120), ("o", 6144), ("mlp", 5120), ("down", 17408)];
pub fn tag_k(tag: &str) -> usize {
    TAGS.iter().find(|t| t.0 == tag).unwrap().1
}

/// Consumer weights (short name, HFQ suffix after `layers.{L}.`) for a layer type and tag.
pub fn consumers(layer_type: &str, tag: &str) -> Vec<(&'static str, &'static str)> {
    let gdn = layer_type == "linear_attention";
    match (tag, gdn) {
        ("attn", true) => vec![
            ("wqkv", "linear_attn.in_proj_qkv.weight"),
            ("wz", "linear_attn.in_proj_z.weight"),
            ("w_beta", "linear_attn.in_proj_b.weight"),
            ("w_alpha", "linear_attn.in_proj_a.weight"),
        ],
        ("attn", false) => vec![
            ("wq", "self_attn.q_proj.weight"),
            ("wk", "self_attn.k_proj.weight"),
            ("wv", "self_attn.v_proj.weight"),
        ],
        ("o", true) => vec![("wo", "linear_attn.out_proj.weight")],
        ("o", false) => vec![("wo", "self_attn.o_proj.weight")],
        ("mlp", _) => vec![
            ("w_gate", "mlp.gate_proj.weight"),
            ("w_up", "mlp.up_proj.weight"),
        ],
        ("down", _) => vec![("w_down", "mlp.down_proj.weight")],
        _ => panic!("bad tag {tag}"),
    }
}

pub struct Capture {
    pub dir: PathBuf,
    pub t: usize,
    pub s1: Vec<f32>,
    pub s2: Vec<f32>,
    pub manifest: Option<Value>,
    pub manifest_text: String,
}

fn read_f32(path: &Path) -> Vec<f32> {
    let len = fs::metadata(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .len() as usize;
    assert_eq!(len % 4, 0);
    let mut v = vec![0f32; len / 4];
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, len) };
    fs::File::open(path).unwrap().read_exact(bytes).unwrap();
    v
}

impl Capture {
    pub fn open(dir: &str) -> Capture {
        let dir = PathBuf::from(dir);
        let manifest_text = fs::read_to_string(dir.join("manifest.json")).unwrap_or_default();
        let manifest: Option<Value> = serde_json::from_str(&manifest_text).ok();
        let s1 = read_f32(&dir.join("signs1.f32"));
        let s2 = read_f32(&dir.join("signs2.f32"));
        assert!(
            s1.len() >= 256 && s2.len() >= 256,
            "sign tables shorter than 256"
        );
        let (s1, s2) = (s1[..256].to_vec(), s2[..256].to_vec());
        let mut t = manifest
            .as_ref()
            .and_then(|m| m.get("T"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        if t == 0 {
            t = fs::metadata(dir.join("tokens.u32"))
                .map(|m| m.len() as usize / 4)
                .unwrap_or(0);
        }
        assert!(t > 0 && t % 2 == 0, "cannot determine T");
        Capture {
            dir,
            t,
            s1,
            s2,
            manifest,
            manifest_text,
        }
    }
    /// Layers/tags present on disk, sorted.
    pub fn entries(&self) -> Vec<(usize, String)> {
        let mut v = Vec::new();
        for e in fs::read_dir(&self.dir).unwrap() {
            let name = e.unwrap().file_name().to_string_lossy().to_string();
            if let Some(rest) = name
                .strip_prefix('L')
                .and_then(|r| r.strip_suffix(".xrot.f32"))
            {
                if let Some((l, tag)) = rest.split_once('.') {
                    if let Ok(l) = l.parse::<usize>() {
                        v.push((l, tag.to_string()));
                    }
                }
            }
        }
        v.sort();
        v
    }
    pub fn load_tag(&self, layer: usize, tag: &str) -> TagData {
        let k = tag_k(tag);
        let t = self.t;
        let nb = k / 128;
        let xrot = read_f32(&self.dir.join(format!("L{layer}.{tag}.xrot.f32")));
        assert_eq!(xrot.len(), t * k, "xrot size L{layer}.{tag}");
        let a4 = fs::read(self.dir.join(format!("L{layer}.{tag}.a4.bin"))).unwrap();
        assert_eq!(a4.len(), nb * t * 72, "a4.bin size L{layer}.{tag}");
        let mut d = vec![0f32; t * nb];
        let mut q = vec![0i8; t * k];
        d.par_chunks_mut(nb)
            .zip(q.par_chunks_mut(k))
            .enumerate()
            .for_each(|(tok, (dr, qr))| {
                for e in 0..nb {
                    let rec = &a4[(e * t + tok) * 72..(e * t + tok + 1) * 72];
                    dr[e] = f32::from_le_bytes(rec[0..4].try_into().unwrap());
                    for j in 0..64 {
                        let b = rec[8 + j];
                        let (lo, hi) = ((b & 15) as i8, (b >> 4) as i8);
                        qr[e * 128 + 2 * j] = if lo >= 8 { lo - 16 } else { lo };
                        qr[e * 128 + 2 * j + 1] = if hi >= 8 { hi - 16 } else { hi };
                    }
                }
            });
        TagData { k, t, xrot, d, q }
    }
}

pub struct TagData {
    pub k: usize,
    pub t: usize,
    pub xrot: Vec<f32>,
    pub d: Vec<f32>,
    pub q: Vec<i8>,
}

impl TagData {
    /// r' = x' - d*q (exact f32 via fused multiply-add), token-major [T][K].
    pub fn residual(&self) -> Vec<f32> {
        let k = self.k;
        let nb = k / 128;
        let mut r = vec![0f32; self.t * k];
        r.par_chunks_mut(k).enumerate().for_each(|(tok, out)| {
            for e in 0..nb {
                let d = self.d[tok * nb + e];
                for j in 0..128 {
                    let i = e * 128 + j;
                    out[i] = (-(self.q[tok * k + i] as f32)).mul_add(d, self.xrot[tok * k + i]);
                }
            }
        });
        r
    }
    /// CPU c2 re-quant parity vs the device codes: (frac identical codes, frac identical d, frac fully identical blocks).
    pub fn parity(&self) -> (f64, f64, f64) {
        let k = self.k;
        let nb = k / 128;
        let (c, dd, bb): (u64, u64, u64) = (0..self.t)
            .into_par_iter()
            .map(|tok| {
                let (mut c, mut dd, mut bb) = (0u64, 0u64, 0u64);
                let mut q = [0i8; 128];
                for e in 0..nb {
                    let x = &self.xrot[tok * k + e * 128..tok * k + (e + 1) * 128];
                    let d = quant::quantize_block(x, &mut q);
                    let dev = &self.q[tok * k + e * 128..tok * k + (e + 1) * 128];
                    let same = (0..128).filter(|&i| q[i] == dev[i]).count() as u64;
                    c += same;
                    let dsame = d.to_bits() == self.d[tok * nb + e].to_bits();
                    dd += dsame as u64;
                    bb += (dsame && same == 128) as u64;
                }
                (c, dd, bb)
            })
            .reduce(|| (0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        let blocks = (self.t * nb) as f64;
        (
            c as f64 / (blocks * 128.0),
            dd as f64 / blocks,
            bb as f64 / blocks,
        )
    }
}

// ---------- synthetic capture ----------
struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.unit()).sum::<f64>() - 6.0
    }
}

/// Write a synthetic capture (device a4.bin layout, c2 quantizer) for the given layers and all four tags.
pub fn synth(dir: &str, layers: &[usize], t: usize) {
    fs::create_dir_all(dir).unwrap();
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let signs: Vec<f32> = (0..512)
        .map(|_| if rng.unit() > 0.5 { 1.0 } else { -1.0 })
        .collect();
    let wf = |p: &str, v: &[f32]| {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        fs::write(Path::new(dir).join(p), b).unwrap();
    };
    wf("signs1.f32", &signs[..256]);
    wf("signs2.f32", &signs[256..]);
    fs::write(Path::new(dir).join("tokens.u32"), vec![0u8; t * 4]).unwrap();
    for &l in layers {
        for (tag, k) in TAGS {
            // heavy-tailed per-channel scale in the awq domain, a few outlier channels
            let chan: Vec<f64> = (0..k)
                .map(|i| (rng.normal() * 0.5).exp() * if i % 211 == 7 { 12.0 } else { 1.0 })
                .collect();
            let mut x = vec![0f32; t * k];
            for tok in 0..t {
                let amp = (rng.normal() * 0.3).exp();
                for i in 0..k {
                    x[tok * k + i] = (rng.normal() * chan[i] * amp) as f32;
                }
            }
            x.par_chunks_mut(k)
                .for_each(|row| rot::rotate_fwd(row, &signs[..256], &signs[256..]));
            wf(&format!("L{l}.{tag}.xrot.f32"), &x);
            let nb = k / 128;
            let mut a4 = vec![0u8; nb * t * 72];
            a4.par_chunks_mut(72).enumerate().for_each(|(idx, rec)| {
                let (e, tok) = (idx / t, idx % t);
                let mut q = [0i8; 128];
                let d =
                    quant::quantize_block(&x[tok * k + e * 128..tok * k + (e + 1) * 128], &mut q);
                rec[0..4].copy_from_slice(&d.to_le_bytes());
                rec[4..8]
                    .copy_from_slice(&(q.iter().map(|v| *v as i32).sum::<i32>()).to_le_bytes());
                for j in 0..64 {
                    rec[8 + j] = ((q[2 * j] as u8) & 15) | (((q[2 * j + 1] as u8) & 15) << 4);
                }
            });
            fs::write(Path::new(dir).join(format!("L{l}.{tag}.a4.bin")), a4).unwrap();
        }
    }
    fs::write(
        Path::new(dir).join("manifest.json"),
        serde_json::to_string_pretty(&json!({"synthetic": true, "T": t, "layers": layers}))
            .unwrap(),
    )
    .unwrap();
}
