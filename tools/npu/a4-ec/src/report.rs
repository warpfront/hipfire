// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Family pooling, NPU cost model (ESTIMATE), KLD allocation (ESTIMATE), greedy ranking, verdict, tables.
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fmt::Write;

pub const M_TOK: f64 = 8192.0;
pub const NPU_TOPS: f64 = 26.9e12;
pub const BETA: f64 = 0.853;
pub const PORT_BPS: f64 = 55e9;
pub const V8_BYTES_PER_OP: f64 = 2.05e-3; // 2.05 B/kop
pub const LAUNCH_FULL_MS: f64 = 0.215;
pub const LAUNCH_LEAN_MS: f64 = 0.050;
pub const HANDOFF_MS: f64 = 0.02;
pub const GAP_KLD: f64 = 0.027;
pub const BUDGETS: [f64; 5] = [100.0, 250.0, 500.0, 1000.0, 2000.0];

pub struct Family {
    pub id: &'static str,
    pub label: &'static str,
    pub calls: f64,
    pub gpu_ms: f64,
    pub n: f64,
    pub k: f64,
}
pub const FAMS: [Family; 7] = [
    Family {
        id: "gdn_qkv",
        label: "GDN qkv",
        calls: 48.0,
        gpu_ms: 11.463,
        n: 10240.0,
        k: 5120.0,
    },
    Family {
        id: "gdn_zba",
        label: "GDN z+b+a",
        calls: 48.0,
        gpu_ms: 7.604,
        n: 6400.0,
        k: 5120.0,
    },
    Family {
        id: "fa_q",
        label: "FA q(+gate)",
        calls: 16.0,
        gpu_ms: 13.722,
        n: 12288.0,
        k: 5120.0,
    },
    Family {
        id: "fa_kv",
        label: "FA k/v",
        calls: 32.0,
        gpu_ms: 1.386,
        n: 1024.0,
        k: 5120.0,
    },
    Family {
        id: "o_proj",
        label: "o_proj",
        calls: 64.0,
        gpu_ms: 6.706,
        n: 5120.0,
        k: 6144.0,
    },
    Family {
        id: "gate_up",
        label: "gate_up",
        calls: 64.0,
        gpu_ms: 38.547,
        n: 34816.0,
        k: 5120.0,
    },
    Family {
        id: "down",
        label: "down",
        calls: 64.0,
        gpu_ms: 18.939,
        n: 5120.0,
        k: 17408.0,
    },
];
pub fn family_of(weight: &str) -> &'static str {
    match weight {
        "wqkv" => "gdn_qkv",
        "wz" | "w_beta" | "w_alpha" => "gdn_zba",
        "wq" => "fa_q",
        "wk" | "wv" => "fa_kv",
        "wo" => "o_proj",
        "w_gate" | "w_up" => "gate_up",
        "w_down" => "down",
        _ => panic!("weight {weight}"),
    }
}

#[derive(Clone, Debug)]
pub enum CostKind {
    StaticA8(f64),
    DynA8(f64),
    FullA8,
    Outlier(f64),
    LowRank(usize, bool),
}

#[derive(Clone)]
pub struct Method {
    pub id: String,
    pub label: String,
    pub cost: Option<CostKind>,
    pub path: Vec<String>,
    pub mode: Mode,
}
#[derive(Clone, Copy)]
pub enum Mode {
    Resid(&'static str),
    LowRank,
    FullK,
}

fn ranking_p(r: &str, p: &str, field: &'static str) -> Method {
    Method {
        id: String::new(),
        label: String::new(),
        cost: None,
        path: vec!["rankings".into(), r.into(), p.into()],
        mode: Mode::Resid(field),
    }
}

pub fn methods() -> Vec<Method> {
    let mut v = Vec::new();
    for p in ["5", "10", "20", "40"] {
        let pf: f64 = p.parse::<f64>().unwrap() / 100.0;
        let mut m = ranking_p("static_calib", p, "resid_exact");
        m.id = format!("static_exact_{p}");
        m.label = format!("static top-{p}% blocks, exact correction (bound, not implementable)");
        v.push(m);
        let mut m = ranking_p("static_insample", p, "resid_exact");
        m.id = format!("insample_exact_{p}");
        m.label = format!("static in-sample top-{p}%, exact correction (optimistic)");
        v.push(m);
        let mut m = ranking_p("static_calib", p, "resid_a8");
        m.id = format!("static_a8_{p}");
        m.label = format!("static A8 residual plane on top-{p}% blocks");
        m.cost = Some(CostKind::StaticA8(pf));
        v.push(m);
        let mut m = ranking_p("dyn_oracle", p, "resid_a8");
        m.id = format!("dynoracle_a8_{p}");
        m.label = format!("dynamic-oracle A8 on top-{p}% blocks/token (NPU full-K plane)");
        m.cost = Some(CostKind::DynA8(pf));
        v.push(m);
        let mut m = ranking_p("dyn_proxy", p, "resid_a8");
        m.id = format!("dynproxy_a8_{p}");
        m.label = format!("dynamic-proxy A8 on top-{p}% blocks/token (NPU full-K plane)");
        m.cost = Some(CostKind::DynA8(pf));
        v.push(m);
        let mut m = ranking_p("dyn_oracle", p, "resid_exact");
        m.id = format!("dynoracle_exact_{p}");
        m.label = format!("dynamic-oracle top-{p}% blocks/token, exact correction (bound)");
        v.push(m);
        let mut m = ranking_p("dyn_proxy", p, "resid_exact");
        m.id = format!("dynproxy_exact_{p}");
        m.label = format!("dynamic-proxy top-{p}% blocks/token, exact correction (bound)");
        v.push(m);
    }
    v.push(Method {
        id: "fullk_a8".into(),
        label: "full-K A8 residual plane (ceiling)".into(),
        cost: Some(CostKind::FullA8),
        path: vec!["fullk_a8".into()],
        mode: Mode::FullK,
    });
    for (dom, name) in [("outlier_rot", "rotated"), ("outlier_orig", "original")] {
        for c in ["0.5", "1", "2", "5", "20"] {
            let cf: f64 = c.parse::<f64>().unwrap() / 100.0;
            v.push(Method {
                id: format!(
                    "{}_{c}",
                    if dom == "outlier_rot" {
                        "outrot"
                    } else {
                        "outorig"
                    }
                ),
                label: format!("outlier channels {c}% K, {name} domain, exact correction"),
                cost: Some(CostKind::Outlier(cf)),
                path: vec!["outlier".into(), dom.into(), c.into()],
                mode: Mode::Resid("resid"),
            });
        }
    }
    for k in [16usize, 32, 64] {
        for (key, name, idp) in [
            ("lowrank_da", "data-aware", "lr_da"),
            ("lowrank_df", "data-free", "lr_df"),
        ] {
            for split in [false, true] {
                v.push(Method {
                    id: format!("{idp}_{k}{}", if split { "_split" } else { "" }),
                    label: format!(
                        "low-rank U(V r') rank {k} {name}{}",
                        if split {
                            " (split: NPU p=V q, GPU applies U p)"
                        } else {
                            ""
                        }
                    ),
                    cost: Some(CostKind::LowRank(k, split)),
                    path: vec![key.into(), k.to_string()],
                    mode: Mode::LowRank,
                });
            }
        }
    }
    v
}

fn get<'a>(v: &'a Value, path: &[String]) -> &'a Value {
    let mut c = v;
    for p in path {
        c = &c[p.as_str()];
    }
    c
}
/// (resid_energy, total_energy) of a method on one case.
fn resid_etot(m: &Method, case: &Value) -> (f64, f64) {
    let node = get(case, &m.path);
    match m.mode {
        Mode::Resid(f) => (node[f].as_f64().unwrap(), node["e_tot"].as_f64().unwrap()),
        Mode::FullK => (
            node["resid"].as_f64().unwrap(),
            node["e_tot"].as_f64().unwrap(),
        ),
        Mode::LowRank => {
            let e = case["e_eval"].as_f64().unwrap();
            ((1.0 - node["f"].as_f64().unwrap()) * e, e)
        }
    }
}

#[derive(Clone)]
pub struct Cost {
    pub ops: f64,
    pub rd: f64,
    pub wr_bf16: f64,
    pub wr_f32: f64,
    pub npu_ms: f64,
    pub npu_ms_f32: f64,
    pub npu_ms_lean: f64,
    pub bound: &'static str,
    pub gpu_ms: f64,
    pub gpu_extra_split_ms: f64,
}

pub fn cost(kind: &CostKind, f: &Family) -> Cost {
    let (m, n, k) = (M_TOK, f.n, f.k);
    let nb = (k / 128.0) as usize;
    let sel = |p: f64| (((p * nb as f64).round() as usize).clamp(1, nb) as f64) / nb as f64;
    let (ops, rd_min, wr_b, wr_f, gpu, gpu_x): (f64, f64, f64, f64, f64, f64) = match kind {
        CostKind::StaticA8(p) | CostKind::Outlier(p) => {
            let ks = sel(*p) * k;
            (
                2.0 * m * n * ks,
                m * ks + n * ks,
                m * n * 2.0,
                m * n * 4.0,
                sel(*p) * f.gpu_ms,
                0.0,
            )
        }
        CostKind::DynA8(p) => (
            2.0 * m * n * k,
            m * k + n * k,
            m * n * 2.0,
            m * n * 4.0,
            sel(*p) * f.gpu_ms,
            0.0,
        ),
        CostKind::FullA8 => (
            2.0 * m * n * k,
            m * k + n * k,
            m * n * 2.0,
            m * n * 4.0,
            f.gpu_ms,
            0.0,
        ),
        CostKind::LowRank(r, split) => {
            let r = (*r as f64).min(n).min(k);
            let g = (r / k + r / n) * f.gpu_ms;
            if *split {
                (
                    2.0 * m * k * r,
                    m * k + r * k,
                    m * r * 4.0,
                    m * r * 4.0,
                    g,
                    r / n * f.gpu_ms,
                )
            } else {
                (
                    2.0 * m * r * (k + n),
                    m * k + r * k + n * r,
                    m * n * 2.0,
                    m * n * 4.0,
                    g,
                    0.0,
                )
            }
        }
    };
    let rd = rd_min.max(ops * V8_BYTES_PER_OP);
    let t_ops = ops / (NPU_TOPS * BETA);
    let ms = |wr: f64, launch: f64| {
        let tb = (rd + wr) / PORT_BPS;
        (
            t_ops.max(tb) * 1e3 + launch + HANDOFF_MS,
            if tb > t_ops { "bandwidth" } else { "compute" },
        )
    };
    let (a, bound) = ms(wr_b, LAUNCH_FULL_MS);
    let (b, _) = ms(wr_f, LAUNCH_FULL_MS);
    let (c, _) = ms(wr_b, LAUNCH_LEAN_MS);
    Cost {
        ops,
        rd,
        wr_bf16: wr_b,
        wr_f32: wr_f,
        npu_ms: a,
        npu_ms_f32: b,
        npu_ms_lean: c,
        bound,
        gpu_ms: gpu,
        gpu_extra_split_ms: gpu_x,
    }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

pub struct Out {
    pub results: Value,
    pub tables: String,
}

pub fn build(cases: &[Value], meta: Value) -> Out {
    let ms = methods();
    // group cases by family -> layer
    let mut by: BTreeMap<&str, BTreeMap<u64, Vec<&Value>>> = BTreeMap::new();
    for c in cases {
        by.entry(family_of(c["weight"].as_str().unwrap()))
            .or_default()
            .entry(c["layer"].as_u64().unwrap())
            .or_default()
            .push(c);
    }
    // pooled f per (family, method) = mean over layers of (1 - sum resid / sum etot)
    let pooled = |fam: &str, m: &Method| -> Option<f64> {
        let layers = by.get(fam)?;
        let fs: Vec<f64> = layers
            .values()
            .map(|cs| {
                let (mut r, mut e) = (0.0, 0.0);
                for c in cs {
                    let (a, b) = resid_etot(m, c);
                    r += a;
                    e += b;
                }
                1.0 - r / e
            })
            .collect();
        Some(mean(&fs))
    };
    // NSR and error-energy weights
    let mut fam_info = serde_json::Map::new();
    let mut w_nsr: BTreeMap<&str, f64> = BTreeMap::new();
    let mut w_abs: BTreeMap<&str, f64> = BTreeMap::new();
    for f in FAMS.iter() {
        if let Some(layers) = by.get(f.id) {
            let mut nsrs = vec![];
            let mut abse = vec![];
            for cs in layers.values() {
                let e: f64 = cs.iter().map(|c| c["e_eval"].as_f64().unwrap()).sum();
                let s: f64 = cs.iter().map(|c| c["s_eval"].as_f64().unwrap()).sum();
                nsrs.push(e / s);
                abse.push(e / (cs[0]["t"].as_f64().unwrap() / 2.0));
            }
            let nsr = mean(&nsrs);
            let ae = mean(&abse);
            w_nsr.insert(f.id, f.calls * nsr);
            w_abs.insert(f.id, f.calls * ae);
            fam_info.insert(f.id.into(), json!({"calls": f.calls, "layers": layers.keys().collect::<Vec<_>>(), "mean_nsr": nsr, "mean_err_energy_per_token": ae}));
        }
    }
    let (sn, sa): (f64, f64) = (w_nsr.values().sum(), w_abs.values().sum());
    for f in FAMS.iter() {
        if let Some(i) = fam_info.get_mut(f.id) {
            i["w_nsr"] = json!(w_nsr[f.id] / sn);
            i["w_abs"] = json!(w_abs[f.id] / sa);
            i["kld_alloc_estimate"] = json!(GAP_KLD * w_nsr[f.id] / sn);
        }
    }
    let wn = |id: &str| w_nsr.get(id).map(|v| v / sn).unwrap_or(0.0);
    let wa = |id: &str| w_abs.get(id).map(|v| v / sa).unwrap_or(0.0);

    // recovery table + options
    let mut rec = serde_json::Map::new();
    let mut options: Vec<Value> = Vec::new();
    for m in &ms {
        let mut per = serde_json::Map::new();
        let mut wmean = 0.0;
        let mut wmean_abs = 0.0;
        for f in FAMS.iter() {
            if let Some(x) = pooled(f.id, m) {
                per.insert(f.id.into(), json!(x));
                wmean += wn(f.id) * x;
                wmean_abs += wa(f.id) * x;
                if let Some(ck) = &m.cost {
                    let c = cost(ck, f);
                    options.push(json!({"family": f.id, "method": m.id, "f": x, "dkld_estimate": GAP_KLD * wn(f.id) * x,
                    "npu_ms_per_call": c.npu_ms, "npu_ms_step": c.npu_ms * f.calls, "npu_ms_step_f32out": c.npu_ms_f32 * f.calls, "npu_ms_step_lean": c.npu_ms_lean * f.calls,
                    "gpu_alone_ms_step": c.gpu_ms * f.calls, "gpu_extra_split_ms_step": c.gpu_extra_split_ms * f.calls, "gemm_ms_step": f.gpu_ms * f.calls,
                    "ops": c.ops, "bytes_read": c.rd, "bytes_written_bf16": c.wr_bf16, "bytes_written_f32": c.wr_f32, "bound": c.bound,
                    "npu_ms_per_call_f32out": c.npu_ms_f32, "npu_ms_per_call_lean": c.npu_ms_lean, "label": "ESTIMATE"}));
                }
            }
        }
        rec.insert(m.id.clone(), json!({"label": m.label, "per_family": Value::Object(per), "weighted_mean_nsr_w": wmean, "weighted_mean_abs_w": wmean_abs}));
    }

    // greedy over (family, method) with incremental upgrades
    let opt_f = |o: &Value, k: &str| o[k].as_f64().unwrap();
    let greedy = |budget: f64| -> (Vec<Value>, f64, f64, BTreeMap<String, String>) {
        let mut cur: BTreeMap<String, (f64, f64, String)> = BTreeMap::new(); // family -> (ms, dkld, method)
        let (mut spent, mut gain) = (0.0, 0.0);
        let mut steps = Vec::new();
        loop {
            let mut best: Option<(f64, &Value, f64, f64)> = None;
            for o in &options {
                let fam = o["family"].as_str().unwrap().to_string();
                let (cm, cd) = cur.get(&fam).map(|c| (c.0, c.1)).unwrap_or((0.0, 0.0));
                let (ms_, d) = (opt_f(o, "npu_ms_step"), opt_f(o, "dkld_estimate"));
                let (dm, dd) = (ms_ - cm, d - cd);
                if dd <= 1e-12 || spent + dm > budget {
                    continue;
                }
                let ratio = if dm <= 1e-9 { f64::INFINITY } else { dd / dm };
                if best.map_or(true, |b| ratio > b.0) {
                    best = Some((ratio, o, dm, dd));
                }
            }
            let Some((ratio, o, dm, dd)) = best else {
                break;
            };
            let fam = o["family"].as_str().unwrap().to_string();
            cur.insert(
                fam.clone(),
                (
                    opt_f(o, "npu_ms_step"),
                    opt_f(o, "dkld_estimate"),
                    o["method"].as_str().unwrap().to_string(),
                ),
            );
            spent += dm;
            gain += dd;
            steps.push(json!({"family": fam, "method": o["method"], "incr_npu_ms": dm, "incr_dkld": dd, "ratio_dkld_per_ms": ratio, "cum_npu_ms": spent, "cum_dkld": gain}));
        }
        (
            steps,
            spent,
            gain,
            cur.into_iter().map(|(k, v)| (k, v.2)).collect(),
        )
    };
    let (order, _, _, _) = greedy(f64::INFINITY);
    let mut curves = Vec::new();
    for b in BUDGETS {
        let (_, spent, gain, asg) = greedy(b);
        curves.push(json!({"budget_npu_ms": b, "spent_npu_ms": spent, "dkld_estimate": gain, "kld_after_estimate": 0.069 - gain, "assignment": asg}));
    }
    let mut singles: Vec<&Value> = options.iter().collect();
    singles.sort_by(|a, b| {
        (opt_f(b, "dkld_estimate") / opt_f(b, "npu_ms_step"))
            .partial_cmp(&(opt_f(a, "dkld_estimate") / opt_f(a, "npu_ms_step")))
            .unwrap()
    });

    // verdict
    let mstat = ms.iter().find(|m| m.id == "static_exact_20").unwrap();
    let mut fs20 = serde_json::Map::new();
    for f in FAMS.iter() {
        if let Some(x) = pooled(f.id, mstat) {
            fs20.insert(f.id.into(), json!(x));
        }
    }
    let mean_nsr_w: f64 = FAMS
        .iter()
        .filter_map(|f| pooled(f.id, mstat).map(|x| wn(f.id) * x))
        .sum();
    let mean_abs_w: f64 = FAMS
        .iter()
        .filter_map(|f| pooled(f.id, mstat).map(|x| wa(f.id) * x))
        .sum();
    let verdict = if mean_nsr_w < 0.5 {
        "KILL selective-A8 route (mean f_static(20%) < 0.5)"
    } else {
        "SURVIVES kill rule (mean f_static(20%) >= 0.5)"
    };
    let verdict_v = json!({"f_static_20_per_family": Value::Object(fs20), "mean_weighted_by_calls_x_nsr": mean_nsr_w, "mean_weighted_by_abs_error_energy": mean_abs_w,
        "threshold": 0.5, "verdict": verdict, "kill": mean_nsr_w < 0.5, "weights_note": "family weight = calls x mean NSR (normalised); absolute-energy variant = calls x mean per-token error energy"});

    let results = json!({"meta": meta, "label_cost_kld": "ESTIMATE", "families": Value::Object(fam_info.clone()), "recovery": Value::Object(rec.clone()),
        "cost_options": options, "greedy_order": order, "budget_curves": curves, "verdict": verdict_v, "cases": cases});

    // ---------------- tables.md ----------------
    let mut md = String::new();
    let pf = |x: f64| format!("{:.3}", x);
    writeln!(md, "# A4 error-correction concentration study\n").unwrap();
    writeln!(md, "Cost and KLD sections are ESTIMATES (model arithmetic on measured constants). Concentration/recovery tables are measured on the capture.\n").unwrap();
    writeln!(md, "## Verdict\n\n**{}**\n", verdict).unwrap();
    writeln!(
        md,
        "| family | f_static(20%) exact (eval) | weight calls*NSR |\n|---|---:|---:|"
    )
    .unwrap();
    for f in FAMS.iter() {
        if let Some(x) = pooled(f.id, mstat) {
            writeln!(md, "| {} | {} | {:.3} |", f.label, pf(x), wn(f.id)).unwrap();
        }
    }
    writeln!(
        md,
        "| **mean (NSR-weighted)** | **{}** | |\n| mean (abs-energy-weighted) | {} | |\n",
        pf(mean_nsr_w),
        pf(mean_abs_w)
    )
    .unwrap();

    {
        let mn =
            |f: &dyn Fn(&Value) -> f64| cases.iter().map(|c| f(c)).fold(f64::INFINITY, f64::min);
        let mx = |f: &dyn Fn(&Value) -> f64| cases.iter().map(|c| f(c)).fold(0.0, f64::max);
        writeln!(md, "## Setup and checks\n").unwrap();
        writeln!(md, "- T = {}, calibration = first T/2 tokens, evaluation = last T/2; {} (layer, tag, weight) cases; blocks of 128; static sets selected on calibration only; dynamic sets per eval token.", meta["T"], cases.len()).unwrap();
        writeln!(md, "- CPU c2 re-quant parity vs device codes: min identical-code fraction {:.6}, min identical-d fraction {:.6} over all (layer, tag).", mn(&|c| c["parity"]["frac_codes"].as_f64().unwrap()), mn(&|c| c["parity"]["frac_d"].as_f64().unwrap())).unwrap();
        writeln!(
            md,
            "- W' r' = W'_o r_o identity (first eval tile, <=64 rows): max relative error {:.2e}.",
            mx(&|c| c["rot_identity_rel_err"].as_f64().unwrap())
        )
        .unwrap();
        writeln!(md, "- Randomized low-rank (oversample 32): recovery at {} vs {} subspace iterations differs by at most {:.2e} (max over all cases, ranks, data-aware/data-free).", meta["lowrank_iters"][0], meta["lowrank_iters"][1], mx(&|c| c["lowrank_conv"]["max_abs_diff_recovery"].as_f64().unwrap())).unwrap();
        writeln!(md, "- Sum over blocks of per-block error energy / total error energy (eval) ranges {:.3}..{:.3}: block errors are nearly uncorrelated, so diag share ~ true recovery.", mn(&|c| c["diag_check"]["sum_g_eval_over_e_eval"].as_f64().unwrap()), mx(&|c| c["diag_check"]["sum_g_eval_over_e_eval"].as_f64().unwrap())).unwrap();
        writeln!(md, "- `*_exact` and outlier methods are exact-f32 corrections (upper bounds); only `*_a8_*`, `fullk_a8`, low-rank are priced as implementable int planes. Original-domain outlier correction would additionally need an unrotated copy of the weight columns (W'_o). Rank k >= N saturates (N=48 beta/alpha). Data-aware low-rank is fit on only T/2 calibration samples and loses to data-free V (population SVD of W') on eval because r' is close to white.\n").unwrap();
    }
    writeln!(md, "## 1. Concentration per layer x tag x weight (eval half; static sets chosen on calib half)\n").unwrap();
    writeln!(md, "diag = share of eval error energy sum_b E_b inside the set; f = true recovery with exact correction of the set. NSR = E/S (eval). parity = CPU c2 re-quant identical codes / identical d.\n").unwrap();
    writeln!(md, "| L | tag | weight | NxK | NSR | parity codes/d | diag 5/10/20/40% | f_static 5/10/20/40% | f_insample 20% | f_dynOracle 20% | f_dynProxy 20% |\n|---|---|---|---|---:|---|---|---|---:|---:|---:|").unwrap();
    for c in cases {
        let r = &c["rankings"];
        let g = |rk: &str, p: &str, f: &str| r[rk][p][f].as_f64().unwrap();
        let ps = ["5", "10", "20", "40"];
        writeln!(
            md,
            "| {} | {} | {} | {}x{} | {:.4} | {:.4}/{:.4} | {} | {} | {} | {} | {} |",
            c["layer"],
            c["tag"].as_str().unwrap(),
            c["weight"].as_str().unwrap(),
            c["n"],
            c["k"],
            c["nsr_eval"].as_f64().unwrap(),
            c["parity"]["frac_codes"].as_f64().unwrap_or(f64::NAN),
            c["parity"]["frac_d"].as_f64().unwrap_or(f64::NAN),
            ps.iter()
                .map(|p| pf(g("static_calib", p, "diag_share")))
                .collect::<Vec<_>>()
                .join("/"),
            ps.iter()
                .map(|p| pf(g("static_calib", p, "f_exact")))
                .collect::<Vec<_>>()
                .join("/"),
            pf(g("static_insample", "20", "f_exact")),
            pf(g("dyn_oracle", "20", "f_exact")),
            pf(g("dyn_proxy", "20", "f_exact"))
        )
        .unwrap();
    }
    writeln!(
        md,
        "\nDynamic sets, diag share at 20% (oracle / proxy) and A8 recoveries per case:\n"
    )
    .unwrap();
    writeln!(md, "| L | tag | weight | diag dynO/dynP 20% | f_A8 static 5/10/20/40% | f_A8 dynO 20% | f_A8 dynP 20% | f_A8 full-K | rot-identity rel err | lowrank conv max|d| |\n|---|---|---|---|---|---|---|---:|---:|---:|").unwrap();
    for c in cases {
        let r = &c["rankings"];
        let g = |rk: &str, p: &str, f: &str| r[rk][p][f].as_f64().unwrap();
        let ps = ["5", "10", "20", "40"];
        writeln!(
            md,
            "| {} | {} | {} | {}/{} | {} | {} | {} | {} | {:.1e} | {:.1e} |",
            c["layer"],
            c["tag"].as_str().unwrap(),
            c["weight"].as_str().unwrap(),
            pf(g("dyn_oracle", "20", "diag_share")),
            pf(g("dyn_proxy", "20", "diag_share")),
            ps.iter()
                .map(|p| pf(g("static_calib", p, "f_a8")))
                .collect::<Vec<_>>()
                .join("/"),
            pf(g("dyn_oracle", "20", "f_a8")),
            pf(g("dyn_proxy", "20", "f_a8")),
            pf(c["fullk_a8"]["f_a8"].as_f64().unwrap()),
            c["rot_identity_rel_err"].as_f64().unwrap(),
            c["lowrank_conv"]["max_abs_diff_recovery"].as_f64().unwrap()
        )
        .unwrap();
    }

    writeln!(md, "\n## 2. Recovery per method (pooled per family: mean over captured layers of 1 - sum resid / sum E)\n").unwrap();
    write!(md, "| method |").unwrap();
    for f in FAMS.iter() {
        write!(md, " {} |", f.label).unwrap();
    }
    writeln!(md, " wmean (calls*NSR) |").unwrap();
    write!(md, "|---|").unwrap();
    for _ in FAMS.iter() {
        write!(md, "---:|").unwrap();
    }
    writeln!(md, "---:|").unwrap();
    for m in &ms {
        let r = &rec[&m.id];
        write!(md, "| `{}` {} |", m.id, m.label).unwrap();
        for f in FAMS.iter() {
            match r["per_family"][f.id].as_f64() {
                Some(x) => write!(md, " {} |", pf(x)).unwrap(),
                None => write!(md, " - |").unwrap(),
            }
        }
        writeln!(md, " {} |", pf(r["weighted_mean_nsr_w"].as_f64().unwrap())).unwrap();
    }

    writeln!(
        md,
        "\n## 3. NPU cost per method per family at pp8192 (ESTIMATE)\n"
    )
    .unwrap();
    writeln!(md, "M=8192; NPU ms/call = max(ops/(26.9e12*0.853), (bytes_read+bytes_written)/55e9) + {} ms full submit + {} ms ring handoff; bytes_read = max(explicit operand bytes, 2.05 B/kop * ops). Fold/scale handling of the int32 partials is NOT modelled (assumed free). *step* columns multiply by calls/step. GPU-alone = same correction run on the GPU (selective ~ p x GEMM ms, low-rank ~ (k/K+k/N) x GEMM ms; dynamic-set GPU time assumes an optimistic gather). Split low-rank: NPU computes p=V q_lo, GPU applies U p (GPU extra column).\n", LAUNCH_FULL_MS, HANDOFF_MS).unwrap();
    writeln!(md, "| family | method | f | GOP/call | rd MB | wr MB bf16 | NPU ms/call (bf16 / f32out / lean) | bound | NPU ms/step | GPU-alone ms/step | GPU extra (split) ms/step | GEMM ms/step | dKLD |\n|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|").unwrap();
    for f in FAMS.iter() {
        for o in options.iter().filter(|o| o["family"] == f.id) {
            writeln!(md, "| {} | `{}` | {} | {:.1} | {:.0} | {:.0} | {:.2} / {:.2} / {:.2} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.5} |", f.label, o["method"].as_str().unwrap(), pf(opt_f(o, "f")),
            opt_f(o, "ops") / 1e9, opt_f(o, "bytes_read") / 1e6, opt_f(o, "bytes_written_bf16") / 1e6, opt_f(o, "npu_ms_per_call"), opt_f(o, "npu_ms_per_call_f32out"), opt_f(o, "npu_ms_per_call_lean"),
            o["bound"].as_str().unwrap(), opt_f(o, "npu_ms_step"), opt_f(o, "gpu_alone_ms_step"), opt_f(o, "gpu_extra_split_ms_step"), opt_f(o, "gemm_ms_step"), opt_f(o, "dkld_estimate")).unwrap();
        }
    }

    writeln!(md, "\n## 4. KLD estimate (ESTIMATE)\n").unwrap();
    writeln!(md, "Gap 0.027 = A4 0.069 - fp8 0.042 (WT2, dense 27B), allocated to GEMM families proportional to calls x mean NSR (first order, uniform per-unit-NSR sensitivity assumption). Recovered KLD of (family, method) = 0.027 * w_g * f. NPU budgets count NPU busy ms per pp8192 step (bf16 output, full-submit launch).\n").unwrap();
    writeln!(
        md,
        "| family | calls | mean NSR | w_g | KLD share |\n|---|---:|---:|---:|---:|"
    )
    .unwrap();
    for f in FAMS.iter() {
        if let Some(i) = fam_info.get(f.id) {
            writeln!(
                md,
                "| {} | {} | {:.4} | {:.3} | {:.5} |",
                f.label,
                f.calls,
                i["mean_nsr"].as_f64().unwrap(),
                i["w_nsr"].as_f64().unwrap(),
                i["kld_alloc_estimate"].as_f64().unwrap()
            )
            .unwrap();
        }
    }
    writeln!(md, "\nTop 25 single (family, method) by dKLD per NPU-ms:\n\n| rank | family | method | f | NPU ms/step | dKLD | dKLD per 100 NPU-ms |\n|---|---|---|---:|---:|---:|---:|").unwrap();
    for (i, o) in singles.iter().take(25).enumerate() {
        writeln!(
            md,
            "| {} | {} | `{}` | {} | {:.1} | {:.5} | {:.5} |",
            i + 1,
            o["family"].as_str().unwrap(),
            o["method"].as_str().unwrap(),
            pf(opt_f(o, "f")),
            opt_f(o, "npu_ms_step"),
            opt_f(o, "dkld_estimate"),
            100.0 * opt_f(o, "dkld_estimate") / opt_f(o, "npu_ms_step")
        )
        .unwrap();
    }
    writeln!(md, "\nGreedy upgrade order (incremental dKLD/ms):\n\n| step | family | method | +NPU ms | +dKLD | cum NPU ms | cum dKLD |\n|---|---|---|---:|---:|---:|---:|").unwrap();
    for (i, s) in order.iter().enumerate() {
        writeln!(
            md,
            "| {} | {} | `{}` | {:.1} | {:.5} | {:.1} | {:.5} |",
            i + 1,
            s["family"].as_str().unwrap(),
            s["method"].as_str().unwrap(),
            opt_f(s, "incr_npu_ms"),
            opt_f(s, "incr_dkld"),
            opt_f(s, "cum_npu_ms"),
            opt_f(s, "cum_dkld")
        )
        .unwrap();
    }
    writeln!(md, "\nCumulative curve at NPU budgets per pp8192 step:\n\n| budget ms | spent ms | dKLD recovered | KLD after (0.069 - dKLD) | assignment |\n|---:|---:|---:|---:|---|").unwrap();
    for c in &curves {
        writeln!(
            md,
            "| {} | {:.1} | {:.5} | {:.5} | {} |",
            c["budget_npu_ms"],
            opt_f(c, "spent_npu_ms"),
            opt_f(c, "dkld_estimate"),
            opt_f(c, "kld_after_estimate"),
            c["assignment"]
        )
        .unwrap();
    }
    writeln!(md, "\n## 5. Verdict (kill rule)\n\nmean f_static(20%) (eval, exact correction): NSR-weighted {:.3}, abs-energy-weighted {:.3}, threshold 0.5.\n\n**{}**", mean_nsr_w, mean_abs_w, verdict).unwrap();
    Out {
        results,
        tables: md,
    }
}
