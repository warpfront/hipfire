// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
mod analyze;
mod capture;
mod hfqm;
mod quant;
mod report;
mod rot;
use serde_json::{json, Value};
use std::time::Instant;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("synth") => {
            let dir = arg(&args, "--out").unwrap(); let t: usize = arg(&args, "--t").unwrap_or("256".into()).parse().unwrap();
            let layers: Vec<usize> = arg(&args, "--layers").unwrap_or("2".into()).split(',').map(|s| s.parse().unwrap()).collect();
            capture::synth(&dir, &layers, t); println!("synthetic capture written to {dir}");
        }
        Some("run") => run(&args),
        Some("report") => {
            let inp = arg(&args, "--results").unwrap(); let out = arg(&args, "--out").unwrap();
            let v: Value = serde_json::from_slice(&std::fs::read(&inp).unwrap()).unwrap();
            let cases = v["cases"].as_array().unwrap().clone(); let o = report::build(&cases, v["meta"].clone());
            std::fs::write(format!("{out}/results.json"), serde_json::to_string_pretty(&o.results).unwrap()).unwrap();
            std::fs::write(format!("{out}/tables.md"), o.tables).unwrap();
        }
        _ => eprintln!("usage: a4-ec synth --out DIR [--t N] [--layers L,..] | run --capture DIR --model PATH --out DIR [--layers L,..] [--tags a,b] [--iters 8,24] | report --results F --out DIR"),
    }
}

fn run(args: &[String]) {
    let t_start = Instant::now();
    let cap = capture::Capture::open(&arg(args, "--capture").unwrap());
    let model = arg(args, "--model").unwrap();
    let out = arg(args, "--out").unwrap();
    std::fs::create_dir_all(&out).unwrap();
    let only_layers: Option<Vec<usize>> =
        arg(args, "--layers").map(|s| s.split(',').map(|x| x.parse().unwrap()).collect());
    let only_tags: Option<Vec<String>> =
        arg(args, "--tags").map(|s| s.split(',').map(|x| x.to_string()).collect());
    let iters: Vec<usize> = arg(args, "--iters")
        .unwrap_or("8,24".into())
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let mut f = std::fs::File::open(&model).unwrap();
    let (mmeta, tensors) = hfqm::index(&mut f);
    let layer_types: Vec<String> = mmeta["config"]["text_config"]["layer_types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let mut cases: Vec<Value> = Vec::new();
    let mut tinfo: Vec<Value> = Vec::new();
    for (layer, tag) in cap.entries() {
        if only_layers.as_ref().map_or(false, |l| !l.contains(&layer))
            || only_tags.as_ref().map_or(false, |t| !t.contains(&tag))
        {
            continue;
        }
        let ltype = layer_types[layer].clone();
        let k = capture::tag_k(&tag);
        let t0 = Instant::now();
        let td = cap.load_tag(layer, &tag);
        let (pc, pd, pb) = td.parity();
        let r = td.residual();
        let ctx = analyze::Ctx::new(td.xrot, r, k, cap.t, &cap.s1, &cap.s2);
        drop(td.d);
        drop(td.q);
        eprintln!(
            "[L{layer} {tag}] loaded+ctx {:.1}s parity codes={pc:.6} d={pd:.6} blocks={pb:.6}",
            t0.elapsed().as_secs_f64()
        );
        for (short, suffix) in capture::consumers(&ltype, &tag) {
            let name = format!("model.language_model.layers.{layer}.{suffix}");
            let ten = tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("tensor {name} missing"))
                .clone();
            assert!(
                ten.qt == 44 && ten.dims.len() == 2 && ten.dims[1] == k,
                "{name}: qt={} dims={:?} (expected qt44 [N,{k}])",
                ten.qt,
                ten.dims
            );
            let in_manifest = cap.manifest_text.contains(&name);
            if !cap.manifest_text.is_empty() && !in_manifest {
                eprintln!("  warn: tensor name {name} not found in capture manifest text");
            }
            tinfo.push(
                json!({"name": name, "qt": ten.qt, "dims": ten.dims, "in_manifest": in_manifest}),
            );
            let w = hfqm::read_dense(&model, &ten);
            let n = ten.dims[0];
            let t1 = Instant::now();
            let mut c = analyze::analyze_weight(&ctx, &w, n, (iters[0], iters[1]));
            c["layer"] = json!(layer);
            c["tag"] = json!(tag);
            c["weight"] = json!(short);
            c["hfq_name"] = json!(name);
            c["layer_type"] = json!(ltype);
            c["parity"] = json!({"frac_codes": pc, "frac_d": pd, "frac_blocks": pb});
            eprintln!("  {short} {n}x{k}: {:.1}s nsr={:.5} f_static20={:.3} f_dynO20={:.3} f_a8_static20={:.3} fullK_a8={:.3} lr_da64={:.3} (conv {:.1e}, ident {:.1e})", t1.elapsed().as_secs_f64(),
                c["nsr_eval"].as_f64().unwrap(), c["rankings"]["static_calib"]["20"]["f_exact"].as_f64().unwrap(), c["rankings"]["dyn_oracle"]["20"]["f_exact"].as_f64().unwrap(),
                c["rankings"]["static_calib"]["20"]["f_a8"].as_f64().unwrap(), c["fullk_a8"]["f_a8"].as_f64().unwrap(), c["lowrank_da"]["64"]["f"].as_f64().unwrap(),
                c["lowrank_conv"]["max_abs_diff_recovery"].as_f64().unwrap(), c["rot_identity_rel_err"].as_f64().unwrap());
            cases.push(c);
            std::fs::write(
                format!("{out}/cases.partial.json"),
                serde_json::to_string(&json!({"meta": {}, "cases": cases})).unwrap(),
            )
            .unwrap();
        }
    }
    let wall = t_start.elapsed().as_secs_f64();
    let bin_md5 = std::fs::read(std::env::current_exe().unwrap())
        .map(|b| format!("{:x}", md5::compute(b)))
        .unwrap_or_default();
    let meta = json!({"capture_dir": cap.dir, "model": model, "T": cap.t, "calib_tokens": cap.t / 2, "eval_tokens": cap.t / 2,
        "synthetic": cap.manifest.as_ref().and_then(|m| m.get("synthetic")).is_some(), "layer_types_total": layer_types.len(), "tensors": tinfo,
        "wall_secs": wall, "binary_md5": bin_md5, "lowrank_iters": iters, "block": 128, "static_p": analyze::PS, "outlier_c": analyze::CS,
        "capture_manifest_model_sha256": cap.manifest.as_ref().and_then(|m| m["model"]["sha256"].as_str().map(String::from)), "capture_manifest_md5": format!("{:x}", md5::compute(cap.manifest_text.as_bytes())),
        "tokens_split": "calibration = first T/2 tokens, evaluation = last T/2 tokens"});
    let o = report::build(&cases, meta);
    let rj = serde_json::to_string_pretty(&o.results).unwrap();
    std::fs::write(format!("{out}/results.json"), &rj).unwrap();
    std::fs::write(format!("{out}/tables.md"), &o.tables).unwrap();
    let _ = std::fs::remove_file(format!("{out}/cases.partial.json"));
    println!(
        "wall {wall:.1}s binary_md5 {bin_md5} results_md5 {:x}",
        md5::compute(rj.as_bytes())
    );
}
