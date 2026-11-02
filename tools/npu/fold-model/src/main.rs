// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
mod activations;
mod hfqm;
use fold_contract_model::{bf_fold, current, fixed, grid, group4, reference};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs::File};
const NAMES: [&str; 5] = ["ief15", "ief16", "bf16mac", "group4", "once"];
const SEED: u64 = 0x6a09e667f3bcc909;
#[derive(Clone, Default)]
struct Stat {
    n: usize,
    rel: f64,
    maxrel: f64,
    rel64: f64,
    maxrel64: f64,
    abs: f64,
    maxabs: f64,
    sq: f64,
    sq64: f64,
    ref_sq: f64,
    ref_abs: f64,
    zero: usize,
    zero64: usize,
    zero_mismatch: usize,
    current_sq: f64,
}
impl Stat {
    fn add(&mut self, y: f32, old: f32, rf: f64) {
        let error = (y as f64 - old as f64).abs();
        let error64 = (y as f64 - rf).abs();
        self.n += 1;
        self.abs += error;
        self.maxabs = self.maxabs.max(error);
        if old != 0.0 {
            let rel = error / (old as f64).abs();
            self.rel += rel;
            self.maxrel = self.maxrel.max(rel);
        } else {
            self.zero += 1;
            if y != 0.0 {
                self.zero_mismatch += 1;
            }
        }
        if rf != 0.0 {
            let rel = error64 / rf.abs();
            self.rel64 += rel;
            self.maxrel64 = self.maxrel64.max(rel);
        } else {
            self.zero64 += 1;
        }
        self.sq += error * error;
        self.sq64 += error64 * error64;
        self.ref_sq += rf * rf;
        self.ref_abs += rf.abs();
        self.current_sq += (old as f64 - rf).powi(2);
    }
    fn merge(&mut self, x: &Self) {
        self.n += x.n;
        self.rel += x.rel;
        self.maxrel = self.maxrel.max(x.maxrel);
        self.rel64 += x.rel64;
        self.maxrel64 = self.maxrel64.max(x.maxrel64);
        self.abs += x.abs;
        self.maxabs = self.maxabs.max(x.maxabs);
        self.sq += x.sq;
        self.sq64 += x.sq64;
        self.ref_sq += x.ref_sq;
        self.ref_abs += x.ref_abs;
        self.zero += x.zero;
        self.zero64 += x.zero64;
        self.zero_mismatch += x.zero_mismatch;
        self.current_sq += x.current_sq;
    }
    fn json(&self) -> Value {
        json!({"n":self.n,"mean_rel_current":self.rel/(self.n-self.zero) as f64,"max_rel_current":self.maxrel,
            "mean_rel_f64":self.rel64/(self.n-self.zero64) as f64,"max_rel_f64":self.maxrel64,
            "mean_abs":self.abs/self.n as f64,"max_abs":self.maxabs,"rel_l1_current":self.abs/self.ref_abs,
            "nrmse_current":(self.sq/self.ref_sq).sqrt(),"nrmse_f64":(self.sq64/self.ref_sq).sqrt(),
            "current_nrmse_f64":(self.current_sq/self.ref_sq).sqrt(),"zero_current":self.zero,"zero_mismatch":self.zero_mismatch})
    }
}
fn stats_json(stats: &[Stat]) -> Value {
    json!(NAMES
        .iter()
        .zip(stats)
        .map(|(n, s)| (n.to_string(), s.json()))
        .collect::<serde_json::Map<_, _>>())
}
fn save(path: &str, value: &Value) {
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert!(
        args.len() >= 2,
        "fold-contract-model MODEL [OUTPUT [CAPTURE_DIR]] or MODEL --audit LAYER_REPORT_DIR OUTPUT"
    );
    let path = &args[1];
    let mut file = File::open(path).unwrap();
    let (meta, tensors) = hfqm::index(&mut file);
    if args.len() == 2 {
        println!(
            "{}",
            json!({"path":path,"config":meta.get("config"),"symmetric":meta.get("mq4v2.symmetric")})
        );
        for t in tensors {
            println!(
                "{}",
                json!({"name":t.name,"qt":t.qt,"shape":t.dims,"offset":t.off,"size":t.size})
            );
        }
        return;
    }
    if args[2] == "--audit" {
        let result = hfqm::audit(&mut file, &tensors, &args[3]);
        save(&args[4], &result);
        println!("{}", result);
        return;
    }
    let capture = args.get(3);
    let mut reports = Vec::new();
    let mut layers: BTreeMap<String, Vec<Stat>> = BTreeMap::new();
    let mut kinds: BTreeMap<String, Vec<Stat>> = BTreeMap::new();
    let mut total = vec![Stat::default(); 5];
    let (mut nscale, mut loss15, mut loss16) = (0usize, 0usize, 0usize);
    let mut sampled_bytes = 0;
    let mut hash = 0xcbf29ce484222325u64;
    let mut act_hash = 0xcbf29ce484222325u64;
    for t in tensors.iter().filter(|t| {
        (t.qt == 44 || t.qt == 53)
            && t.name.contains(".layers.")
            && !t.name.starts_with("mtp.")
            && (t.dims.len() == 2 || t.name.contains(".mlp.experts."))
    }) {
        let k = *t.dims.last().unwrap();
        if k % 128 != 0 {
            continue;
        }
        let n = t.dims[t.dims.len() - 2];
        let experts = if t.dims.len() == 3 { t.dims[0] } else { 1 };
        let ne = experts.min(8);
        let nr = n.min(32);
        assert_eq!(t.size as usize, experts * n * (k / 128 * 68));
        let layer = t
            .name
            .split(".layers.")
            .nth(1)
            .unwrap()
            .split('.')
            .next()
            .unwrap()
            .to_owned();
        let kind = t
            .name
            .split(".layers.")
            .nth(1)
            .unwrap()
            .split_once('.')
            .unwrap()
            .1
            .to_owned();
        let mut stats = vec![Stat::default(); 5];
        for ei in 0..ne {
            let expert = ei * experts / ne;
            let acts = if let Some(dir) = capture.filter(|_| experts > 1) {
                let gu = activations::captured(dir, layer.parse().unwrap(), expert);
                if t.name.ends_with("down_proj") {
                    let gu_name = t.name.replace("down_proj", "gate_up_proj");
                    let gate = tensors.iter().find(|t| t.name == gu_name).unwrap();
                    activations::down(&mut file, gate, expert, &gu)
                } else {
                    gu
                }
            } else {
                (0..8).map(|i| activations::synthetic(k, i, SEED)).collect()
            };
            for (d, q) in &acts {
                for byte in d
                    .iter()
                    .flat_map(|x| x.to_bits().to_le_bytes())
                    .chain(q.iter().map(|x| *x as u8))
                {
                    act_hash ^= byte as u64;
                    act_hash = act_hash.wrapping_mul(0x100000001b3);
                }
            }
            for ri in 0..nr {
                let row = ri * n / nr;
                let bytes = hfqm::row(&mut file, t, expert, row);
                sampled_bytes += bytes.len();
                for &byte in &bytes {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(0x100000001b3);
                }
                let (scales, weights) = hfqm::decode(t, &bytes);
                for (bits, loss) in [(15, &mut loss15), (16, &mut loss16)] {
                    let (s, a) = grid(&scales, bits);
                    for (&v, &s) in scales.iter().zip(&s) {
                        if s as f64 * 2f64.powi(a) != v as f64 {
                            *loss += 1;
                        }
                    }
                }
                nscale += scales.len();
                for act in &acts {
                    let c = activations::dot(&scales, &weights, act);
                    let old = current(&scales, &act.0, &c);
                    let rf = reference(&scales, &act.0, &c);
                    let once = ((scales.iter().map(|v| *v as f64).sum::<f64>()
                        / scales.len() as f64)
                        * (act.0.iter().map(|v| *v as f64).sum::<f64>() / act.0.len() as f64)
                        * c.iter().sum::<i32>() as f64) as f32;
                    let ys = [
                        fixed(&scales, &act.0, &c, 15),
                        fixed(&scales, &act.0, &c, 16),
                        bf_fold(&scales, &act.0, &c),
                        group4(&scales, &act.0, &c),
                        once,
                    ];
                    for (s, y) in stats.iter_mut().zip(ys) {
                        s.add(y, old, rf);
                    }
                }
            }
        }
        let ls = layers
            .entry(layer)
            .or_insert_with(|| vec![Stat::default(); 5]);
        let ks = kinds
            .entry(kind)
            .or_insert_with(|| vec![Stat::default(); 5]);
        for i in 0..5 {
            ls[i].merge(&stats[i]);
            ks[i].merge(&stats[i]);
            total[i].merge(&stats[i]);
        }
        reports.push(json!({"tensor":t.name,"shape":t.dims,"experts_sampled":ne,"rows_per_expert":nr,"contracts":stats_json(&stats)}));
    }
    let result = json!({"path":path,"file_bytes":file.metadata().unwrap().len(),"capture_dir":capture,
        "activation_source":if capture.is_some() {"Flash experts: real routed captured normed x, CPU FWHT256 and producer c2; down propagated from canonical current-fold BF16 gate/up via host exp SwiGLU, FWHT128/c2. Other tensors synthetic rotated domain. CPU producer is not device-byte certified."} else {"synthetic CLT normal rotated-domain, epoch amplitude exp(0.4*N); correct c2 candidate indices5/7; 8 tokens; unchanged q"},
        "seed":SEED,"sample_weight_fnv64":format!("{hash:016x}"),"sample_activation_fnv64":format!("{act_hash:016x}"),"sampled_bytes":sampled_bytes,
        "weight_scales":nscale,"weight_scale_loss_15":loss15,"weight_scale_loss_16":loss16,"total":stats_json(&total),
        "layers":layers.iter().map(|(l,s)|(l.to_owned(),stats_json(s))).collect::<serde_json::Map<_,_>>(),
        "gemm_kinds":kinds.iter().map(|(l,s)|(l.to_owned(),stats_json(s))).collect::<serde_json::Map<_,_>>(),"tensors":reports});
    save(&args[2], &result);
    println!(
        "{}",
        json!({"path":path,"tensors":reports.len(),"sampled_bytes":sampled_bytes,"weight_scale_loss_15":loss15,"weight_scale_loss_16":loss16,"total":result["total"]})
    );
}
