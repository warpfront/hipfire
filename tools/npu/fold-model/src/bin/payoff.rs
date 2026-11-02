// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
use serde_json::json;
// Measured Halo pp8192 calls and kernel time: docs/dense27b.md.
const SHAPES: [(&str, usize, usize, usize, f64); 7] = [
    ("qkv", 10240, 5120, 48, 11.463),
    ("zba", 6400, 5120, 48, 7.604),
    ("fa_q", 12288, 5120, 16, 13.722),
    ("fa_kv", 1024, 5120, 32, 1.386),
    ("o", 5120, 6144, 64, 6.706),
    ("gate_up", 34816, 5120, 64, 38.547),
    ("down", 5120, 17408, 64, 18.939),
];
fn slowdown(gbps: f64) -> f64 {
    // Unreliable Halo curve, 4.13% within-session drift. Sensitivity only.
    if gbps <= 5.3 {
        1.0 + 0.022 * gbps / 5.3
    } else if gbps <= 16.0 {
        1.022 + (gbps - 5.3) / (16.0 - 5.3) * (0.059 - 0.022)
    } else {
        1.059 + (gbps - 16.0) / (46.9 - 16.0) * (0.113 - 0.059)
    }
}
fn main() {
    let g0 = SHAPES.iter().map(|s| s.3 as f64 * s.4).sum::<f64>();
    let non = 6810.69 - g0;
    let mut all = Vec::new();
    for (arch, ratio, non_ratio) in [
        ("gfx1151 Halo", 1.0, 1.0),
        ("gfx1152 860M memory-scaled remainder", 0.225, 2.0),
        ("gfx1152 860M compute-scaled remainder", 0.225, 1.0 / 0.225),
    ] {
        for rate in [19.0, 21.0, 23.0, 26.9, 30.2] {
            for (mode, gpu_loss) in [
                ("mixed-current-gpu-ief15-npu", 1.0),
                ("full-ief15-transferred-g0-penalty", 1.57),
            ] {
                for bytes_kop in [2.44, 3.05] {
                    let mut rows = Vec::new();
                    let mut coop = 0.0;
                    let mut weighted_share = 0.0;
                    let mut ops_all = 0.0;
                    for &(name, n, k, calls, g_ms) in &SHAPES {
                        let ops = 2.0 * 8192.0 * n as f64 * k as f64 / 1e12;
                        let mut best = (g_ms / ratio * gpu_loss, 0.0, 0.0);
                        // Current V2B admission needs256rows per gate/up projection.
                        let granule = if name == "gate_up" { 512 } else { 256 };
                        for cols in (granule..n).step_by(granule) {
                            let f = cols as f64 / n as f64;
                            let output_bytes = 8192.0 * cols as f64 * 4.0;
                            // Measured repeated-tile traffic, not one-time host operand bytes.
                            let busy_ms = (ops * f / (rate * 0.853) * 1000.0)
                                .max(ops * f * bytes_kop * 1e9 / 55e6);
                            let traffic = ops * f * bytes_kop * 1e9 / busy_ms / 1e6;
                            let n_ms = busy_ms + 0.215;
                            let g = g_ms / ratio * (1.0 - f) * gpu_loss * slowdown(traffic);
                            let producer =
                                ((k / 128) as f64 * 8192.0 * 6.0 + 8192.0 * 2.0) / 200e6 + 0.004;
                            let epi = output_bytes * 2.0 / 200e6;
                            let step = g.max(n_ms) + producer + epi + 0.020;
                            if step < best.0 {
                                best = (step, f, traffic);
                            }
                        }
                        coop += best.0 * calls as f64;
                        weighted_share += ops * best.1 * calls as f64;
                        ops_all += ops * calls as f64;
                        rows.push(json!({"kind":name,"fraction":best.1,"call_ms":best.0,"npu_gbps":best.2}));
                    }
                    let alone = g0 / ratio + non * non_ratio;
                    let elapsed = coop + non * non_ratio;
                    all.push(json!({"arch":arch,"contract_mode":mode,"gpu_compute_ratio":ratio,"npu_raw_tops_assumed":rate,"npu_beta_assumed":0.853,
                "gpu_fold_ratio":gpu_loss,"npu_bytes_per_kop":bytes_kop,"baseline_ms":alone,"coop_ms":elapsed,"gain_pct":100.0*(alone/elapsed-1.0),
                "optimal_ops_share":weighted_share/ops_all,"gemms":rows}));
                }
            }
        }
    }
    let result=serde_json::to_string_pretty(&json!({"measured_halo_gemm_ms":g0,"measured_halo_remainder_ms":non,
        "note":"ALL gains/shares ESTIMATES. Mixed mode keeps current GPU fold; split is numeric identity. Full mode transfers gfx1151 measuredGPU1.57 penalty to gfx1152, not a gfx1152 measurement. NPU beta/GPU slowdown copied from drift-flaggedHalo; V8traffic2.44B/kop or25%retile3.05;55GB/s cap;submit0.215ms;GPUring0.020ms. gfx1152 assumes8CU3GHz,70%IU4efficiency,128GB/sLPDDR5X,identicalNPUrate/port. Metadata/epilogue200GB/s assumed. Actual NPUIEF15/retile traffic and currentfoldstridedGPU speed unmeasured.","rows":all})).unwrap();
    if let Some(path) = std::env::args().nth(1) {
        std::fs::write(path, result).unwrap();
    } else {
        println!("{result}");
    }
}
