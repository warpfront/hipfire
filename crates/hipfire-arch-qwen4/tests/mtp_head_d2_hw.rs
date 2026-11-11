// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native MTP head D2 (`HIPFIRE_QWEN4_MTP_HEAD_D2`) wiring on a real
//! Flash-Next model: trunk D2 observables are unchanged by the head arm (T4),
//! an armed head draft step traces `QWEN4_CPU_ARM rows=1` while an unarmed one
//! traces none (T6), and an armed head under `graphs.capture_mode` returns an
//! error instead of the GPU route (T7).
//!
//! `#[ignore]`d: it needs a real HIP GPU, a discrete card with host-placed
//! experts and the artifact named by `HIPFIRE_MTP_HEAD_D2_MODEL`, plus a fixed
//! `HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=N` in the environment (the trunk D2 is
//! forced on with `HIPFIRE_QWEN4_D2_DEFAULT=1`). Each arm is a fresh child
//! process because the knobs are read from a process-start snapshot.
//!
//! ```text
//! HIPFIRE_MTP_HEAD_D2_MODEL=/path/to/qwen3.8-flash-next.mq4-xts \
//! HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS=5 \
//! cargo test -p hipfire-arch-qwen4 --features reference-parity \
//!     --test mtp_head_d2_hw -- --ignored --exact mtp_head_d2_hw --nocapture
//! ```

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const MODEL_ENV: &str = "HIPFIRE_MTP_HEAD_D2_MODEL";
const SUMMARY_PREFIX: &str = "MTP_HEAD_D2_SUMMARY ";
const MAX_SEQ: usize = 4096;

fn model_path() -> PathBuf {
    let path = std::env::var_os(MODEL_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{MODEL_ENV} must name a Flash-Next artifact"));
    assert!(
        path.is_file(),
        "{MODEL_ENV}={} is not a file",
        path.display()
    );
    path
}

fn run_arm(head_d2: &str, dump: &Path) -> (Value, String) {
    let exe = std::env::current_exe().expect("test binary path");
    let output = Command::new(exe)
        .args([
            "--exact",
            "mtp_head_d2_arm",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(MODEL_ENV, model_path())
        .env("HIPFIRE_QWEN4_D2_DEFAULT", "1")
        .env("HIPFIRE_QWEN4_MTP_HEAD_D2", head_d2)
        .env("HIPFIRE_CPU_SPLIT_TRACE", "1")
        .env("HIPFIRE_MTP_HEAD_D2_DUMP", dump)
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|e| panic!("spawn arm {head_d2}: {e}"));
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "arm {head_d2} failed ({})\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    let lines: Vec<&str> = stdout
        .lines()
        .filter_map(|line| line.split_once(SUMMARY_PREFIX).map(|(_, json)| json))
        .collect();
    assert_eq!(
        lines.len(),
        1,
        "arm {head_d2}: expected one summary\n{stdout}"
    );
    (
        serde_json::from_str(lines[0]).expect("summary json"),
        stderr,
    )
}

fn cpu_arm_lines(stderr: &str) -> Vec<&str> {
    stderr
        .lines()
        .filter(|l| l.contains("QWEN4_CPU_ARM"))
        .collect()
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_HEAD_D2_MODEL"]
fn mtp_head_d2_hw() {
    assert!(
        std::env::var_os("HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS").is_some(),
        "set a fixed HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS so the trunk has host-placed experts"
    );
    let dir = std::env::var_os("HIPFIRE_MTP_HEAD_D2_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("HIPFIRE_MTP_HEAD_D2_OUT must name an output directory"));
    std::fs::create_dir_all(&dir).expect("output dir");
    let (off_dump, on_dump) = (dir.join("logits-off.f32"), dir.join("logits-on.f32"));
    let (off, off_err) = run_arm("0", &off_dump);
    let (on, on_err) = run_arm("1", &on_dump);

    // The trunk partition must be live for the head to follow it.
    let trunk = &off["trunk_after_attach_mtp"];
    assert!(
        trunk["split_layers"]
            .as_array()
            .is_some_and(|l| !l.is_empty()),
        "trunk D2 not armed; the head arm cannot be exercised: {off}"
    );
    // T4: head arming leaves every trunk D2 observable unchanged, within an
    // arm (attach_mtp) and across arms.
    for arm in [&off, &on] {
        assert_eq!(
            arm["trunk_before_attach_mtp"],
            arm["trunk_after_attach_mtp"]
        );
    }
    assert_eq!(off["trunk_after_attach_mtp"], on["trunk_after_attach_mtp"]);
    assert_eq!(on["trunk_after_attach_mtp"]["d2_storage_capacity_rows"], 4);
    assert_eq!(off["head_d2_armed"], false);
    assert_eq!(on["head_d2_armed"], true);
    assert_eq!(on["head_d2_host_bytes"][0], 112_760);
    assert!(off_err.contains("qwen4 MTP head D2: off"), "{off_err}");
    assert!(
        on_err.contains("qwen4 MTP head D2: armed (rows=1"),
        "{on_err}"
    );

    // T6: wiring truth. Armed head step arms the CPU with one row; unarmed none.
    assert_eq!(off["eager_ok"], true, "{off}");
    assert_eq!(on["eager_ok"], true, "{on}");
    assert!(
        cpu_arm_lines(&off_err).is_empty(),
        "unarmed head printed a CPU arm"
    );
    let arm = cpu_arm_lines(&on_err);
    assert!(!arm.is_empty(), "armed head printed no QWEN4_CPU_ARM line");
    assert!(arm.iter().all(|l| l.contains("rows=1")), "{arm:?}");

    // T7: capture is refused for an armed head, never a GPU fallback.
    assert_eq!(on["capture_ok"], false, "{on}");
    assert!(on["capture_error"].is_string());

    // T5: draft-only numerics. The D2 route moves head logits at ULP level;
    // record it (top-1 agreement, max-abs, ULP histogram) instead of assuming.
    // The executor's rows == CPU oracle bytes parity is the trunk harness's
    // (`qwen4_cpu_split_parity --untiered`); this arm records A-vs-B drift only.
    let floats = |path: &Path| -> Vec<f32> {
        std::fs::read(path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };
    let (a, b) = (floats(&off_dump), floats(&on_dump));
    assert_eq!(a.len(), b.len());
    assert!(!a.is_empty());
    let (mut max_abs, mut hist) = (0f32, [0usize; 6]);
    for (x, y) in a.iter().zip(&b) {
        max_abs = max_abs.max((x - y).abs());
        let ulp = (x.to_bits() as i64 - y.to_bits() as i64).unsigned_abs();
        let bin = match ulp {
            0 => 0,
            1 => 1,
            2..=4 => 2,
            5..=16 => 3,
            17..=256 => 4,
            _ => 5,
        };
        hist[bin] += 1;
    }
    let agree = off["parity_top1"]
        .as_array()
        .unwrap()
        .iter()
        .zip(on["parity_top1"].as_array().unwrap())
        .filter(|(x, y)| x == y)
        .count();
    println!(
        "MTP_HEAD_D2_PARITY steps={} top1_agree={agree} max_abs={max_abs} ulp_hist[0,1,2-4,5-16,17-256,>256]={hist:?}",
        off["parity_steps"]
    );
    assert!(max_abs.is_finite());
}

/// One arm in a fresh process; spawned by `mtp_head_d2_hw`.
#[test]
#[ignore = "child process of mtp_head_d2_hw"]
fn mtp_head_d2_arm() {
    let result =
        hipfire_arch_qwen4::state_parity::run_mtp_head_d2_probe(Path::new(&model_path()), MAX_SEQ)
            .unwrap_or_else(|e| panic!("head D2 probe: {e}"));
    println!("{SUMMARY_PREFIX}{result}");
}
