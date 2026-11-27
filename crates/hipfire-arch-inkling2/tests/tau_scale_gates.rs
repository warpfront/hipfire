// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! U4 gate: full-layer tau scaling on beta's single-row vehicle.
//!
//! - Gate A (below floor, real drive): 4-token drive pos 0..3, argmax pinned
//!   to the U3 beta record (83762/16045/118/72), then a capture at the first
//!   full visit with tau ON vs `HIPFIRE_INKLING2_TAU_OFF=1`: every key
//!   bitwise identical (the helper returns `None` below the floor, so the
//!   OFF switch is inert there by construction AND by measurement).
//! - Gate B (past floor, synthetic pos): beta's single-row `drive()` cannot
//!   really reach pos 8192 (per-step device growth walls ~pos 1500-1700 on
//!   this tip — TAU6 census; do NOT attempt a real 8192-deep single-row
//!   drive). Instead: fresh upload, spoof the position state
//!   (`seq_pos`/tags/`visit_cap`) to POS=9216 with zero-filled KV (correct
//!   tags, empty history — deterministic across arms), one single-row step
//!   with capture at the first full visit. Arm selected by `TAU_GATE_ARM`
//!   (`on` default, `off` sets the kill switch); each arm dumps its capture
//!   to `TAU_GATE_DIR/arm_{on,off}/` in a FRESH process. The committed
//!   `tau_past_floor_compare_dumps` test then requires every pre-scale key
//!   bitwise, `qc_scaled`/`rc_scaled == off * oracle::tau(POS)` within 1e-6,
//!   and equal tail argmaxes. Real-history past-floor proof (blocked prefill
//!   + capture) runs through the blocked-prefill gate.
//!
//! `#[ignore]`d: needs the 14 GB MQ4 artifact + gfx1100/gfx1151. Fresh process each:
//!
//! ```powershell
//! $env:HIPFIRE_HOME = 'D:\hipfire-models\inkling2-home'
//! $env:HIPFIRE_GRAPH = '0'
//! cargo test -p hipfire-arch-inkling2 --test tau_scale_gates -- --ignored --nocapture --test-threads=1
//! $env:TAU_GATE_ARM = 'on';  <run gate A/B on>   # dump below_on / arm_on
//! $env:TAU_GATE_ARM = 'off'; <run gate A/B off>  # dump below_off / arm_off
//! cargo test -p hipfire-arch-inkling2 --test tau_scale_gates compare_dumps -- --ignored --nocapture
//! ```

use hipfire_arch_inkling2::arch_model::{Inkling2Bundle, Inkling2State};
use hipfire_arch_inkling2::config::Inkling2Config;
use hipfire_arch_inkling2::gpu::{
    swa_slot, swa_window_len, upload, Inkling2GpuState, Inkling2GpuWeights,
};
use hipfire_arch_inkling2::weights::Inkling2Weights;
use hipfire_arch_inkling2::{forward_token_gpu, forward_token_gpu_capture, oracle};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::collections::HashMap;

const ART: &str = r"D:\hipfire-models\inkling2-mini-mq4.hfq";
const MAX_SEQ: usize = 16384;
/// Synthetic past-floor position for gate B (matches the TAU6 blocked-vehicle
/// probe so the oracle t is directly comparable: t(9216) = 1.0117892).
const POS: usize = 9216;
/// Pre-scale keys: captured before the tau site, must be bitwise across arms.
const PRE_KEYS: &[&str] = &[
    "x_in", "q_inp", "k_inp", "v_inp", "k_raw", "v_raw", "k_conv", "v_conv", "qc", "rc", "kc", "vc",
];

fn ensure_home() {
    if std::env::var("HIPFIRE_HOME").is_err() {
        std::env::set_var("HIPFIRE_HOME", r"D:\hipfire-models\inkling2-home");
    }
    assert_eq!(
        std::env::var("HIPFIRE_GRAPH").as_deref(),
        Ok("0"),
        "gates require HIPFIRE_GRAPH=0"
    );
}

fn test_gpu() -> Option<Gpu> {
    ensure_home();
    let gpu = match Gpu::init() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("skip: no usable GPU ({e:?})");
            return None;
        }
    };
    if !matches!(gpu.arch.as_str(), "gfx1100" | "gfx1151") {
        eprintln!("skip: requires gfx1100/gfx1151, found {}", gpu.arch);
        return None;
    }
    Some(gpu)
}

fn open_artifact() -> (Inkling2Config, Inkling2Weights) {
    let art = std::env::var("INKLING2_MQ4").unwrap_or_else(|_| ART.to_string());
    eprintln!("tau-gate: artifact {art}");
    let hfq = HfqFile::open(std::path::Path::new(&art)).expect("open artifact");
    let cfg = Inkling2Config::from_metadata_json(&hfq.metadata_json).expect("parse config");
    let weights = Inkling2Weights::load(&hfq, &cfg).expect("index weights");
    (cfg, weights)
}

fn upload_bundle(cfg: Inkling2Config, weights: Inkling2Weights, gpu: &mut Gpu) -> Inkling2Bundle {
    let (gw, gs): (Inkling2GpuWeights, Inkling2GpuState) =
        upload(&weights, &cfg, gpu, MAX_SEQ).expect("gpu upload");
    Inkling2Bundle {
        config: cfg,
        weights,
        state: Inkling2State { seq_pos: 0 },
        eos_tok: 200006,
        gpu_weights: Some(gw),
        gpu_state: Some(gs),
    }
}

fn first_full_visit(cfg: &Inkling2Config) -> usize {
    for v in 0..cfg.loop_cfg.n_slot() {
        let phys = cfg.loop_cfg.layer_phys(v);
        if !cfg.swa_layers[phys] {
            return v;
        }
    }
    panic!("no full-attention visit in the loop schedule");
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .expect("nonempty")
        .0
}

fn gate_dir() -> std::path::PathBuf {
    std::env::var_os("TAU_GATE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("hipfire-tau-gates"))
}

/// Gate A arm: below the floor the port must not change numerics (U3 beta
/// pins). Run on/off in fresh processes so the process-config snapshot sees
/// the selected kill-switch value before GPU initialization.
#[test]
#[ignore = "needs the 14 GB MQ4 artifact + gfx1100/gfx1151"]
fn tau_below_floor_dump() {
    let arm = std::env::var("TAU_GATE_ARM").unwrap_or_else(|_| "on".to_string());
    assert!(arm == "on" || arm == "off", "TAU_GATE_ARM must be on|off");
    if arm == "off" {
        std::env::set_var("HIPFIRE_INKLING2_TAU_OFF", "1");
    } else {
        std::env::remove_var("HIPFIRE_INKLING2_TAU_OFF");
    }
    ensure_home();
    let (cfg, weights) = open_artifact();
    let visit = first_full_visit(&cfg);
    eprintln!("tau-gate A/{arm}: capture visit {visit}");
    let Some(mut gpu) = test_gpu() else {
        return;
    };
    let mut bundle = upload_bundle(cfg, weights, &mut gpu);

    // U3 beta record (gate3b): same vehicle, same artifact.
    let pins = [
        (5000u32, 83762usize),
        (12345, 16045),
        (777, 118),
        (31274, 72),
    ];
    for (pos, (tok, want)) in pins.iter().enumerate() {
        let logits = forward_token_gpu(&mut bundle, &mut gpu, *tok, pos).expect("drive");
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "pos {pos}: non-finite"
        );
        let got = argmax(&logits);
        assert_eq!(got, *want, "pos {pos} tok {tok}: argmax drift below floor");
        eprintln!("tau-gate A/{arm}: pos {pos} tok {tok} argmax={got} (pin {want}) ok");
    }
    let cap =
        forward_token_gpu_capture(&mut bundle, &mut gpu, 7, pins.len(), visit).expect("capture");
    let dir = gate_dir().join(format!("below_{arm}"));
    std::fs::create_dir_all(&dir).expect("gate dir");
    for (k, v) in &cap {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(dir.join(format!("{k}.f32")), bytes).expect("dump");
    }
    eprintln!(
        "tau-gate A/{arm}: dumped {} keys to {}",
        cap.len(),
        dir.display()
    );
}

#[test]
#[ignore = "needs tau below-floor on/off dump files"]
fn tau_below_floor_compare_dumps() {
    let root = gate_dir();
    let on = root.join("below_on");
    let off = root.join("below_off");
    let mut names: Vec<String> = std::fs::read_dir(&on)
        .expect("read on dumps")
        .map(|e| {
            e.expect("dir entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    let mut off_names: Vec<String> = std::fs::read_dir(&off)
        .expect("read off dumps")
        .map(|e| {
            e.expect("dir entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    off_names.sort();
    assert_eq!(names, off_names, "capture key sets differ");
    for k in PRE_KEYS {
        assert!(
            names.iter().any(|name| name == &format!("{k}.f32")),
            "pre-scale key {k} missing from capture (gate contract)"
        );
    }
    for name in names {
        let a = read_f32(&on.join(&name));
        let b = read_f32(&off.join(&name));
        assert_eq!(a.len(), b.len(), "key {name} len");
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "key {name}: differs below floor"
        );
    }
    eprintln!("tau-gate A: PASS (all keys bitwise, argmax pins checked in both arms)");
}

/// Gate B: one arm of the past-floor probe. Run twice in fresh processes
/// (`TAU_GATE_ARM=on` then `off`), compare dumps offline.
#[test]
#[ignore = "needs the 14 GB MQ4 artifact + gfx1100/gfx1151"]
fn tau_past_floor_probe_dump() {
    ensure_home();
    let arm = std::env::var("TAU_GATE_ARM").unwrap_or_else(|_| "on".to_string());
    assert!(
        arm == "on" || arm == "off",
        "TAU_GATE_ARM must be on|off, got {arm}"
    );
    if arm == "off" {
        std::env::set_var("HIPFIRE_INKLING2_TAU_OFF", "1");
    } else {
        std::env::remove_var("HIPFIRE_INKLING2_TAU_OFF");
    }
    let (cfg, weights) = open_artifact();
    let t = oracle::tau(POS, cfg.log_floor, cfg.log_alpha);
    eprintln!("tau-gate B/{arm}: POS={POS} oracle t={t}");
    let visit = first_full_visit(&cfg);
    let Some(mut gpu) = test_gpu() else {
        return;
    };
    let mut bundle = upload_bundle(cfg, weights, &mut gpu);

    // Synthetic position state: consistent tags for every visit. Full
    // visits need tags[i]==i over 0..=POS; SWA visits are a 1024-ring
    // holding only the trailing window — spoof exactly the window the
    // gather checks (`swa_window_len`), older slots are unchecked.
    // KV is zero-filled (empty history). Deterministic across arms by
    // construction; the scale is the only allowed difference.
    {
        let gs = bundle.gpu_state.as_mut().unwrap();
        gs.seq_pos = POS;
        gpu.inkling2_gf_set_pos(&gs.pos_dev, POS)
            .expect("set synthetic device position");
        for v in 0..gs.visit_rows.len() {
            let rows = gs.visit_rows[v];
            let n_kv = swa_window_len(POS, rows);
            assert!(n_kv > 0, "visit {v}: empty window");
            if rows > POS {
                assert!(POS < rows, "visit {v}: POS {POS} >= rows {rows}");
            }
            for k in (POS + 1 - n_kv)..=POS {
                gs.visit_pos_tags[v][swa_slot(k, rows)] = k as u32;
            }
            gs.visit_cap[v] = POS;
        }
    }
    bundle.state.seq_pos = POS;

    let cap: HashMap<String, Vec<f32>> =
        forward_token_gpu_capture(&mut bundle, &mut gpu, 0, POS, visit)
            .expect("past-floor capture");
    let tail = forward_token_gpu(&mut bundle, &mut gpu, 0, POS + 1).expect("tail");
    assert!(tail.iter().all(|v| v.is_finite()), "tail non-finite");
    assert_eq!(tail.len(), bundle.config.unpadded_vocab_size);
    eprintln!("tau-gate B/{arm}: tail argmax={} finite ok", argmax(&tail));

    let dir = gate_dir().join(format!("arm_{arm}"));
    std::fs::create_dir_all(&dir).expect("gate dir");
    for (k, v) in &cap {
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(dir.join(format!("{k}.f32")), b).expect("dump");
    }
    std::fs::write(
        dir.join("meta.txt"),
        format!(
            "pos={POS} visit={visit} oracle_t={t} tail_argmax={} tail_len={}\n",
            argmax(&tail),
            tail.len()
        ),
    )
    .expect("meta");
    eprintln!(
        "tau-gate B/{arm}: dumped {} keys to {}",
        cap.len(),
        dir.display()
    );
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read f32 dump");
    assert_eq!(bytes.len() % 4, 0, "{}: truncated f32 word", path.display());
    bytes
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

fn read_meta(path: &std::path::Path) -> HashMap<String, String> {
    let text = std::fs::read_to_string(path).expect("read tau gate meta");
    text.lines()
        .filter_map(|line| line.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// Gate B comparator: run after the on/off dump arms. This is committed so a
/// wrong finite scale cannot pass merely because both dumps were produced.
#[test]
#[ignore = "needs tau on/off dump files from tau_past_floor_probe_dump"]
fn tau_past_floor_compare_dumps() {
    let root = gate_dir();
    let on = root.join("arm_on");
    let off = root.join("arm_off");
    let meta_on = read_meta(&on.join("meta.txt"));
    let meta_off = read_meta(&off.join("meta.txt"));
    assert_eq!(meta_on["pos"], meta_off["pos"], "position differs");
    assert_eq!(meta_on["visit"], meta_off["visit"], "visit differs");
    assert_eq!(
        meta_on["oracle_t"], meta_off["oracle_t"],
        "oracle t differs"
    );
    assert_eq!(
        meta_on["tail_argmax"], meta_off["tail_argmax"],
        "tail argmax differs"
    );
    let t: f32 = meta_on["oracle_t"].parse().expect("oracle t");

    for k in PRE_KEYS {
        let a = read_f32(&on.join(format!("{k}.f32")));
        let b = read_f32(&off.join(format!("{k}.f32")));
        assert_eq!(a.len(), b.len(), "key {k} len");
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "pre-scale key {k} differs"
        );
    }
    for (scaled, base) in [("qc_scaled", "qc"), ("rc_scaled", "rc")] {
        let a = read_f32(&on.join(format!("{scaled}.f32")));
        let b = read_f32(&off.join(format!("{base}.f32")));
        assert_eq!(a.len(), b.len(), "key {scaled} len");
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y * t).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= 1e-6, "key {scaled}: max |on-off*t|={worst:e}");
        eprintln!("tau-gate B: {scaled} max|on-off*t|={worst:e}");
    }
    eprintln!("tau-gate B: PASS (pre-scale bitwise, scaled ratio, tail argmax)");
}
