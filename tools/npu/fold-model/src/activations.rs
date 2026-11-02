// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use crate::hfqm::{self, Tensor};
use fold_contract_model::{bf, current};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};
pub type Acts = (Vec<f32>, Vec<i8>);
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
// Reproduce producer c2={5,7} candidate indices, NOT divisors {5,7}.
// f32 lane-local four-element FMA MSE and XOR reduction order match HIP source.
pub fn quantize(x: &[f32]) -> Acts {
    assert_eq!(x.len() % 128, 0);
    let mut ds = Vec::new();
    let mut qs = Vec::new();
    for chunk in x.chunks(128) {
        let max = chunk.iter().map(|x| x.abs()).fold(0f32, f32::max);
        if max == 0.0 {
            ds.push(1.0);
            qs.extend([0; 128]);
            continue;
        }
        let base = (max / 7.0) * 0.5;
        let mut best = (f32::INFINITY, 0.0, Vec::new());
        for multiplier in [f32::from_bits(0x3fdb6db7), 2.0] {
            let d = base * multiplier;
            let q = chunk
                .iter()
                .map(|x| (*x / d).round_ties_even().clamp(-8.0, 7.0) as i8)
                .collect::<Vec<_>>();
            let mut errors = [0.0f32; 32];
            for lane in 0..32 {
                for j in 0..4 {
                    let i = lane * 4 + j;
                    let error = (-(q[i] as f32)).mul_add(d, chunk[i]);
                    errors[lane] = error.mul_add(error, errors[lane]);
                }
            }
            for delta in [16, 8, 4, 2, 1] {
                let previous = errors;
                for lane in 0..32 {
                    errors[lane] = previous[lane] + previous[lane ^ delta];
                }
            }
            if errors[0] < best.0 {
                best = (errors[0], d, q);
            }
        }
        ds.push(best.1);
        qs.extend(best.2);
    }
    (ds, qs)
}
pub fn synthetic(k: usize, token: usize, seed: u64) -> Acts {
    let mut rng = Rng(seed ^ (token as u64 + 1) * 0x1f123bb5);
    let mut x = Vec::new();
    for _ in 0..k / 128 {
        let amplitude = (rng.normal() * 0.4).exp();
        for _ in 0..128 {
            x.push((rng.normal() * amplitude) as f32);
        }
    }
    quantize(&x)
}
fn signs(mut state: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}
fn rotate(x: &mut [f32], n: usize) {
    let (a, b) = if n == 256 { (42, 1042) } else { (43, 1043) };
    let s1 = signs(a, n);
    let s2 = signs(b, n);
    let scale = if n == 256 {
        0.0625
    } else {
        1.0 / (128.0f32).sqrt()
    };
    for chunk in x.chunks_mut(n) {
        assert_eq!(chunk.len(), n);
        for i in 0..n {
            chunk[i] *= s1[i];
        }
        let mut stride = 1;
        while stride < n {
            for base in (0..n).step_by(stride * 2) {
                for j in 0..stride {
                    let a = chunk[base + j];
                    let b = chunk[base + j + stride];
                    chunk[base + j] = a + b;
                    chunk[base + j + stride] = a - b;
                }
            }
            stride *= 2;
        }
        for i in 0..n {
            chunk[i] *= scale * s2[i];
        }
    }
}
pub fn dot(scales: &[f32], weights: &[i8], acts: &Acts) -> Vec<i32> {
    assert_eq!(scales.len(), acts.0.len());
    weights
        .chunks(128)
        .zip(acts.1.chunks(128))
        .map(|(w, q)| w.iter().zip(q).map(|(&w, &q)| w as i32 * q as i32).sum())
        .collect()
}
pub fn captured(dir: &str, layer: usize, expert: usize) -> Vec<Acts> {
    let ids = std::fs::read(format!("{dir}/L{layer}.ids.bin")).unwrap();
    assert_eq!(ids.len() % 40, 0);
    let matching = ids
        .chunks(40)
        .enumerate()
        .filter(|(_, slots)| {
            slots
                .chunks(4)
                .any(|id| i32::from_le_bytes(id.try_into().unwrap()) == expert as i32)
        })
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    assert!(
        !matching.is_empty(),
        "expert {expert} has no real routed token in layer {layer}"
    );
    let mut file = File::open(format!("{dir}/L{layer}.x.bin")).unwrap();
    assert_eq!(
        file.metadata().unwrap().len(),
        (ids.len() / 40 * 2560 * 4) as u64
    );
    (0..8)
        .map(|i| {
            let token = matching[i * matching.len() / 8];
            let mut bytes = vec![0; 2560 * 4];
            file.seek(SeekFrom::Start((token * bytes.len()) as u64))
                .unwrap();
            file.read_exact(&mut bytes).unwrap();
            let mut x = bytes
                .chunks(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            rotate(&mut x, 256);
            quantize(&x)
        })
        .collect()
}
// Real captured gate/up inputs, propagated through canonical quantized weights.
// BF16 boundaries match the route, but host exp and producer math are NOT a GPU-byte oracle.
pub fn down(file: &mut File, gate: &Tensor, expert: usize, acts: &[Acts]) -> Vec<Acts> {
    assert_eq!(gate.dims[1], 1280);
    let mut hidden = vec![vec![0.0f32; 640]; acts.len()];
    let mut gates = vec![vec![0.0f32; 640]; acts.len()];
    for feature in 0..1280 {
        let bytes = hfqm::row(file, gate, expert, feature);
        let (scales, weights) = hfqm::decode(gate, &bytes);
        for (token, act) in acts.iter().enumerate() {
            let c = dot(&scales, &weights, act);
            let y = bf(current(&scales, &act.0, &c));
            if feature < 640 {
                gates[token][feature] = y;
            } else {
                let g = gates[token][feature - 640];
                hidden[token][feature - 640] = bf((g / (1.0 + (-g).exp())) * y);
            }
        }
    }
    hidden
        .into_iter()
        .map(|mut x| {
            rotate(&mut x, 128);
            quantize(&x)
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn c2_clips_with_correct_candidate_indices() {
        let mut x = vec![0.1f32; 128];
        x[0] = 8.0;
        let (d, q) = quantize(&x);
        assert!(
            (d[0] - 8.0 / 7.0).abs() < 1e-6
                || (d[0] - 8.0 / 7.0 * 0.5 * f32::from_bits(0x3fdb6db7)).abs() < 1e-6
        );
        assert_eq!(q[0], 7);
    }
    #[test]
    fn all_zero_has_producer_scale_one() {
        assert_eq!(quantize(&[0.0; 128]), (vec![1.0], vec![0; 128]));
    }
}
