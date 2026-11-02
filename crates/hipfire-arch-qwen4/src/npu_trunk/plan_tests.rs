// SPDX-License-Identifier: Apache-2.0
//! Python↔Rust cost-model golden vectors, including the original 90 plans.
use super::*;
use hipfire_runtime::coproc::*;
use serde_json::{json, Value};
use std::path::PathBuf;
fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/npu_trunk")
        .join(name)
}
fn cal(name: &str) -> Calibration {
    // Preserve legacy lab fixtures; explicitly adapt identity and zero-cost
    // direct consume so the original 90 cost vectors remain unchanged.
    let mut v: Value = serde_json::from_slice(&std::fs::read(fixture(name)).unwrap()).unwrap();
    v["schema_version"] = json!(CALIBRATION_SCHEMA_VERSION);
    // Python's legacy get('panel_cols', 1), not a production input default.
    if v["npu"].get("panel_cols").is_none() {
        v["npu"]["panel_cols"] = json!(1);
    }
    v["margin"] = json!(0);
    v["key"] = json!({"device":{"arch":"gfx1151","pci_bdf":"0000:c0:00.0","npu_fw_version":"fixture"},"build":"fixture","kernel":{"uname_release":"fixture"},"model_trunk_index_sha256":"0000000000000000000000000000000000000000000000000000000000000000","epilogue":v["npu"]["epilogue"],"trunk_route":"stock"});
    v["glue"]["consume_mode"] = json!("direct");
    v["glue"]["consume_ms_per_col_8192"] = json!({"bf16":2.85/16480.0,"f32":4.60/16480.0});
    for name in ["qkv", "zab", "q", "kvidx"] {
        v["families"][name]["out_dtype"] = json!(if name == "qkv" { "bf16" } else { "f32" });
    }
    let c: Calibration = serde_json::from_value(v).unwrap();
    c.validate(&FLASH_NEXT).unwrap();
    c
}
fn mask(names: &str) -> FamilyMask {
    if names.contains("zab") {
        FamilyMask::ALL
    } else {
        FamilyMask::from(Family::GdnQkv) | FamilyMask::QSA
    }
}
fn golden(names: &str) {
    let vectors: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/npu_trunk/golden.json"
    )))
    .unwrap();
    let mut count = 0;
    for v in vectors
        .as_array()
        .unwrap()
        .iter()
        .filter(|v| v["families"] == names)
    {
        let name = v["file"].as_str().unwrap();
        let c = cal(name);
        let families = mask(names);
        let star = v["n_star"].as_u64().map(|n| n as usize);
        assert_eq!(
            crossover(&FLASH_NEXT, &c, families.into()),
            star,
            "{name} {names}"
        );
        for p in v["plans"].as_array().unwrap() {
            let rows = p["rows"].as_u64().unwrap() as usize;
            let actual = plan(&FLASH_NEXT, &c, families.into(), rows);
            let expected: Vec<_> = p["steps"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| {
                    if let Some(r) = s.get("gpu") {
                        Step::Gpu {
                            chunk: r[0].as_u64().unwrap() as usize..r[1].as_u64().unwrap() as usize,
                        }
                    } else {
                        let r = &s["pipe"];
                        Step::Pipe {
                            a: r[0][0].as_u64().unwrap() as usize
                                ..r[0][1].as_u64().unwrap() as usize,
                            b: r[1][0].as_u64().unwrap() as usize
                                ..r[1][1].as_u64().unwrap() as usize,
                        }
                    }
                })
                .collect();
            assert_eq!(
                actual.steps, expected,
                "{name} {names} rows={rows}: {}",
                p["python_plan"]
            );
            for (value, expected) in [
                (actual.t_pipe_ms, p["t_pipe_ms"].as_f64().unwrap()),
                (actual.t_gpu_ms, p["t_gpu_ms"].as_f64().unwrap()),
            ] {
                assert!(
                    (value - expected).abs() <= expected.abs() * 1e-6,
                    "{name} {names} rows={rows}: {value} != {expected}"
                );
            }
            if name == "g80bf16-measured.json" {
                println!(
                    "families={names} n*={star:?} rows={rows} gpu={:.9} pipe={:.9} {}",
                    actual.t_gpu_ms,
                    actual.t_pipe_ms,
                    request_log_line(&FLASH_NEXT, rows, &actual)
                );
            }
            count += 1;
        }
    }
    assert_eq!(count, 27);
}
#[test]
fn npu_trunk_golden_family_a() {
    golden("qkv,q,kvidx");
}
#[test]
fn npu_trunk_golden_family_b() {
    golden("qkv,zab,q,kvidx");
}
#[test]
fn npu_trunk_margin_and_determinism() {
    let mut c = cal("g80bf16-measured.json");
    let initial = plan(&FLASH_NEXT, &c, FamilyMask::ALL.into(), 8192);
    assert!(matches!(initial.steps[0], Step::Pipe { .. }));
    c.margin = 0.10;
    assert!(matches!(
        plan(&FLASH_NEXT, &c, FamilyMask::ALL.into(), 8192).steps[0],
        Step::Gpu { .. }
    ));
    c.margin = 0.99;
    for n in [0, 2048, 8192, 9319, 16384, 24576, 65536] {
        let p = plan(&FLASH_NEXT, &c, FamilyMask::ALL.into(), n);
        assert_eq!(p, plan(&FLASH_NEXT, &c, FamilyMask::ALL.into(), n));
        assert_eq!(p.t_pipe_ms, p.t_gpu_ms);
        assert!(p.steps.iter().all(|s| matches!(s, Step::Gpu { .. })));
    }
    assert_eq!(crossover(&FLASH_NEXT, &c, FamilyMask::ALL.into()), None);
}
#[test]
fn npu_trunk_key_mismatches_and_logs() {
    let c = cal("g80bf16-measured.json");
    let p = plan(&FLASH_NEXT, &c, FamilyMask::ALL.into(), 16384);
    assert_eq!(c.key_matches(&c.key), Ok(()));
    let mutations: [(&str, fn(&mut CalKey)); 8] = [
        ("device.arch", |k| k.device.arch += "x"),
        ("device.pci_bdf", |k| k.device.pci_bdf += "x"),
        ("device.npu_fw_version", |k| k.device.npu_fw_version += "x"),
        ("build", |k| k.build += "x"),
        ("kernel.uname_release", |k| k.kernel.uname_release += "x"),
        ("model_trunk_index_sha256", |k| {
            k.model_trunk_index_sha256 += "x"
        }),
        ("epilogue", |k| k.epilogue += "x"),
        ("trunk_route", |k| k.trunk_route += "x"),
    ];
    for (field, mutate) in mutations {
        let mut key = c.key.clone();
        mutate(&mut key);
        assert_eq!(
            c.key_matches(&key),
            Err(format!("calibration key mismatch: {field}"))
        );
        let log_key = LoadLogKey {
            current: &key,
            mode: "auto",
            families: FamilyMask::ALL.into(),
        };
        assert_eq!(
            load_log_line(&FLASH_NEXT, &c, &log_key, None, &p, &p),
            format!("npu-trunk: off (calibration key mismatch: {field})")
        );
    }
    let key = LoadLogKey {
        current: &c.key,
        mode: "auto",
        families: FamilyMask::ALL.into(),
    };
    let line = load_log_line(
        &FLASH_NEXT,
        &c,
        &key,
        Some(4736),
        &p,
        &plan(&FLASH_NEXT, &c, FamilyMask::ALL.into(), 65536),
    );
    assert!(
        line.starts_with("npu-trunk: mode=auto contract=bf16 families=qkv,zab,q,kvidx cal_key=")
    );
    assert!(line.contains(" cal_sha256="));
    assert!(line.contains(" n*=4736 T(16384)="));
    assert_eq!(
        request_log_line(&FLASH_NEXT, 16384, &p),
        "npu-trunk: rows=16384 plan=[pipe 0..8192+8192..16384] predicted=-14.7%"
    );
}
#[test]
fn npu_trunk_record_roundtrip_and_validation() {
    let c = cal("g80bf16-measured.json");
    let path =
        std::env::temp_dir().join(format!("hipfire-npu-trunk-cal-{}.json", std::process::id()));
    c.store(&path, &FLASH_NEXT).unwrap();
    let loaded = Calibration::load(&path, &FLASH_NEXT).unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(
        serde_json::to_value(&c).unwrap(),
        serde_json::to_value(loaded).unwrap()
    );
    let mut v = serde_json::to_value(&c).unwrap();
    v.as_object_mut().unwrap().remove("margin");
    assert!(serde_json::from_value::<Calibration>(v).is_err());
    let mut invalid = c.clone();
    invalid.schema_version = 1;
    assert!(invalid.validate(&FLASH_NEXT).is_err());
    invalid = c.clone();
    invalid.npu.panel_cols = 0;
    assert!(invalid.validate(&FLASH_NEXT).is_err());
    invalid = c.clone();
    invalid.npu.tops_corun = f64::NAN;
    assert!(invalid.validate(&FLASH_NEXT).is_err());
}
#[test]
fn npu_trunk_empty_partial_masks_and_padding() {
    let c = cal("g80bf16-measured.json");
    assert_eq!(
        npu_ms(&FLASH_NEXT, &c, 0, 8192, 1),
        npu_ms(&FLASH_NEXT, &c, 0, 8192, 1280)
    );
    assert!(npu_ms(&FLASH_NEXT, &c, 0, 8192, 1281) > npu_ms(&FLASH_NEXT, &c, 0, 8192, 1280));
    for families in [FamilyMask::NONE, FamilyMask::from(Family::GdnZ)] {
        let p = plan(&FLASH_NEXT, &c, families.into(), 16384);
        assert_eq!(p.t_pipe_ms, p.t_gpu_ms);
        assert!(p.steps.iter().all(|s| matches!(s, Step::Gpu { .. })));
    }
    assert!(plan(&FLASH_NEXT, &c, FamilyMask::ALL.into(), 0)
        .steps
        .is_empty());
}

fn grouped_mask(names: &Value) -> FamilyMask {
    names
        .as_array()
        .unwrap()
        .iter()
        .fold(FamilyMask::NONE, |mask, name| {
            mask | match name.as_str().unwrap() {
                "qkv" => FamilyMask::from(Family::GdnQkv),
                "zab" => FamilyMask::from_bits(0x0e),
                "q" => FamilyMask::from(Family::QsaQ),
                "kvidx" => FamilyMask::from_bits(0xe0),
                _ => panic!("unknown golden family"),
            }
        })
}
fn close(actual: f64, expected: &Value) {
    let expected = expected.as_f64().unwrap();
    assert!(
        (actual - expected).abs() <= expected.abs().max(1.0) * 1e-10,
        "{actual} != {expected}"
    );
}
#[test]
fn npu_trunk_v2_consume_and_selection_goldens() {
    let vectors: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/npu_trunk/golden-v2.json"
    )))
    .unwrap();
    for v in vectors.as_array().unwrap() {
        let cal: Calibration = serde_json::from_value(v["calibration"].clone()).unwrap();
        cal.validate(&FLASH_NEXT).unwrap();
        let (families, report) = select_families(&FLASH_NEXT, &cal);
        assert_eq!(
            FamilyMask::from(families),
            grouped_mask(&v["selected"]),
            "{}",
            v["name"]
        );
        assert_eq!(
            FamilyMask::from(report.best_families),
            grouped_mask(&v["best_families"]),
            "{}",
            v["name"]
        );
        assert_eq!(report.subsets.len(), 15);
        for (actual, expected) in report.subsets.iter().zip(v["subsets"].as_array().unwrap()) {
            assert_eq!(
                FamilyMask::from(actual.families),
                grouped_mask(&expected["families"])
            );
            assert_eq!(
                actual.beats_margin_16384,
                expected["beats_margin_16384"].as_bool().unwrap()
            );
            close(actual.mean_relative, &expected["mean_relative"]);
            for i in 0..3 {
                close(actual.t_pipe_ms[i], &expected["t_pipe_ms"][i]);
                close(actual.t_gpu_ms[i], &expected["t_gpu_ms"][i]);
                close(actual.relative[i], &expected["relative"][i]);
            }
        }
        for expected in v["bank_segments"].as_array().unwrap() {
            let actual = bank_segments(
                &FLASH_NEXT,
                &cal,
                grouped_mask(&expected["families"]).into(),
                4096,
                8192,
            );
            for i in 0..2 {
                close(actual.gpu[i], &expected["gpu"][i]);
                close(actual.npu[i], &expected["npu"][i]);
            }
        }
        for family_plans in v["plans"].as_array().unwrap() {
            let mask = grouped_mask(&family_plans["families"]);
            for expected in family_plans["plans"].as_array().unwrap() {
                let actual = plan(
                    &FLASH_NEXT,
                    &cal,
                    mask.into(),
                    expected["rows"].as_u64().unwrap() as usize,
                );
                let steps: Vec<_> = expected["steps"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| {
                        let range = |r: &Value| {
                            r[0].as_u64().unwrap() as usize..r[1].as_u64().unwrap() as usize
                        };
                        if let Some(r) = s.get("gpu") {
                            Step::Gpu { chunk: range(r) }
                        } else {
                            Step::Pipe {
                                a: range(&s["pipe"][0]),
                                b: range(&s["pipe"][1]),
                            }
                        }
                    })
                    .collect();
                assert_eq!(
                    actual.steps, steps,
                    "{} {}",
                    v["name"], expected["python_plan"]
                );
                close(actual.t_pipe_ms, &expected["t_pipe_ms"]);
                close(actual.t_gpu_ms, &expected["t_gpu_ms"]);
            }
        }
    }
}
#[test]
fn npu_trunk_v2_consume_is_gpu_work_on_live_columns() {
    let mut c = cal("g80bf16-measured.json");
    let direct = bank_segments(&FLASH_NEXT, &c, FamilyMask::ALL.into(), 8192, 0);
    c.glue.consume_mode = "retile".into();
    let retile = bank_segments(&FLASH_NEXT, &c, FamilyMask::ALL.into(), 8192, 0);
    let expected_gdn = c.glue.consume_ms_per_col_8192["bf16"] * c.families["qkv"].cols as f64
        + c.glue.consume_ms_per_col_8192["f32"] * c.families["zab"].cols as f64;
    let expected_qsa = c.glue.consume_ms_per_col_8192["f32"]
        * (c.families["q"].cols + c.families["kvidx"].cols) as f64;
    assert!((retile.gpu[0] - direct.gpu[0] - expected_gdn).abs() < 1e-10);
    assert!((retile.gpu[1] - direct.gpu[1] - expected_qsa).abs() < 1e-10);
    assert_eq!(retile.npu, direct.npu);
    let half = bank_segments(&FLASH_NEXT, &c, FamilyMask::ALL.into(), 4096, 0);
    assert!((half.gpu[0] * 2.0 - retile.gpu[0]).abs() < 1e-10);
}
#[test]
fn npu_trunk_v2_rejects_missing_invalid_fields() {
    let c = cal("g80bf16-measured.json");
    for pointer in [
        "/key/trunk_route",
        "/glue/consume_mode",
        "/glue/consume_ms_per_col_8192",
        "/families/qkv/out_dtype",
    ] {
        let mut v = serde_json::to_value(&c).unwrap();
        let (parent, field) = pointer.rsplit_once('/').unwrap();
        v.pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(
            serde_json::from_value::<Calibration>(v).is_err(),
            "{pointer}"
        );
    }
    for (pointer, value) in [
        ("/key/trunk_route", json!("overlay:")),
        ("/key/trunk_route", json!("other")),
        ("/glue/consume_mode", json!("other")),
        ("/glue/consume_ms_per_col_8192/bf16", json!(-1.0)),
        ("/families/qkv/out_dtype", json!("f16")),
    ] {
        let mut v = serde_json::to_value(&c).unwrap();
        *v.pointer_mut(pointer).unwrap() = value;
        assert!(
            serde_json::from_value::<Calibration>(v)
                .unwrap()
                .validate(&FLASH_NEXT)
                .is_err(),
            "{pointer}"
        );
    }
    for dtype in ["bf16", "f32"] {
        let mut invalid = c.clone();
        invalid.glue.consume_ms_per_col_8192.remove(dtype);
        assert!(invalid.validate(&FLASH_NEXT).is_err());
        let mut invalid = c.clone();
        invalid
            .glue
            .consume_ms_per_col_8192
            .insert(dtype.into(), f64::NAN);
        assert!(invalid.validate(&FLASH_NEXT).is_err());
    }
    let mut invalid = c.clone();
    invalid
        .glue
        .consume_ms_per_col_8192
        .insert("f16".into(), 0.0);
    assert!(invalid.validate(&FLASH_NEXT).is_err());
    let mut key = c.key.clone();
    key.epilogue = "other".into();
    key.trunk_route = "overlay:gfx1151-zab-v1".into();
    assert_eq!(
        c.key_matches(&key),
        Err("calibration key mismatch: epilogue".into())
    );
    key.epilogue = c.key.epilogue.clone();
    assert_eq!(
        c.key_matches(&key),
        Err("calibration key mismatch: trunk_route".into())
    );
}

#[test]
fn npu_trunk_runtime_key_is_cheap_deterministic_metadata() {
    let digest = "098bd75fd3a6a5174d06033f2b3ec7febeb7777e7eba4696b64cecca5e3418b6";
    let key = calibration_key(
        "gfx1151",
        "0000:C0:00.0",
        "1.1.2.65\n",
        "7.3.0-rc7-perfopt+\n",
        digest,
        "overlay:gfx1151-zab-v1",
    );
    assert_eq!(key.device.pci_bdf, "0000:c0:00.0");
    assert_eq!(key.device.npu_fw_version, "1.1.2.65");
    assert_eq!(key.kernel.uname_release, "7.3.0-rc7-perfopt+");
    assert_eq!(key.build, "hipfire-0.4.2:npu-trunk-bf16-rowmajor-v1");
    assert_eq!(key.model_trunk_index_sha256, digest);
    assert_eq!(key.trunk_route, "overlay:gfx1151-zab-v1");
    assert_eq!(
        key,
        calibration_key(
            "gfx1151",
            "0000:c0:00.0",
            "1.1.2.65",
            "7.3.0-rc7-perfopt+",
            digest,
            "overlay:gfx1151-zab-v1"
        )
    );
}

#[test]
fn npu_trunk_auto_admission_matrix() {
    let c = Calibration::load(fixture("g80bf16-stock-v2.json"), &FLASH_NEXT).unwrap();
    assert_eq!(
        auto_admission(&FLASH_NEXT, &c, &c.key).map(FamilyMask::from),
        Ok(FamilyMask::ALL)
    );
    let mut wrong = c.key.clone();
    wrong.device.arch = "gfx1201".into();
    assert_eq!(
        auto_admission(&FLASH_NEXT, &c, &wrong).map(FamilyMask::from),
        Err("calibration key mismatch: device.arch".into())
    );
    let mut subset = c.clone();
    subset.gpu.trunk_in_ms_per_8192.insert("zab".into(), 0.0);
    assert_eq!(
        FamilyMask::from(select_families(&FLASH_NEXT, &subset).0),
        FamilyMask::from(Family::GdnQkv) | FamilyMask::QSA
    );
    assert_eq!(
        auto_admission(&FLASH_NEXT, &subset, &subset.key).map(FamilyMask::from),
        Ok(FamilyMask::from(Family::GdnQkv) | FamilyMask::QSA)
    );
    let mut none = c.clone();
    none.npu.tops_corun = 0.001;
    assert_eq!(
        FamilyMask::from(select_families(&FLASH_NEXT, &none).0),
        FamilyMask::NONE
    );
    assert_eq!(
        auto_admission(&FLASH_NEXT, &none, &none.key).map(FamilyMask::from),
        Err("calibration selects no families".into())
    );
}

#[test]
fn npu_trunk_auto_maps_natural_and_rebalanced_pairs() {
    for file in ["g80bf16-stock-v2.json", "g80bf16-overlay-v2.json"] {
        let c = Calibration::load(fixture(file), &FLASH_NEXT).unwrap();
        let star = crossover(&FLASH_NEXT, &c, FamilyMask::ALL.into());
        for (rows, expected) in [
            (8000, vec![]),
            (9000, vec![(0, 9000)]),
            (9319, vec![(0, 9319)]),
            (16343, vec![(0, 16343)]),
            (20000, vec![(0, 16384)]),
            (
                57544,
                vec![(0, 16384), (16384, 16384), (32768, 16384), (49152, 8392)],
            ),
            (
                57556,
                vec![(0, 16384), (16384, 16384), (32768, 16384), (49152, 8404)],
            ),
            (
                65536,
                vec![(0, 16384), (16384, 16384), (32768, 16384), (49152, 16384)],
            ),
        ] {
            let planned = plan(&FLASH_NEXT, &c, FamilyMask::ALL.into(), rows);
            let executable = executable_plan(&FLASH_NEXT, &c, &planned, rows, star);
            let pairs: Vec<_> = executable
                .steps
                .iter()
                .filter_map(|step| match step {
                    Step::Pipe { a, b } => Some((a.start, b.end - a.start)),
                    _ => None,
                })
                .collect();
            assert_eq!(pairs, expected, "{file} {rows}");
            let mut at = 0;
            for step in &executable.steps {
                let (start, end) = match step {
                    Step::Gpu { chunk } => (chunk.start, chunk.end),
                    Step::Pipe { a, b } => {
                        assert_eq!(
                            planned_pair_rows(&executable, a.start),
                            Some(b.end - a.start)
                        );
                        assert_eq!(planned_pair(&executable, a.start), Some((a.len(), b.len())));
                        let minimum = 512.max(c.gpu.min_bank_rows);
                        assert!(a.len().min(b.len()) >= minimum && a.len().max(b.len()) <= CHUNK);
                        // Natural 8192 + B, or the rebalanced end of the request.
                        assert!(
                            a.len() == CHUNK
                                || (a.len() % 64 == 0 && b.end == rows && b.end - a.start > CHUNK)
                        );
                        (a.start, b.end)
                    }
                };
                assert_eq!(start, at);
                at = end;
            }
            assert_eq!(at, rows);
        }
        let p = plan(&FLASH_NEXT, &c, FamilyMask::ALL.into(), 16384);
        for cutoff in [None, Some(16385)] {
            assert!(executable_plan(&FLASH_NEXT, &c, &p, 16384, cutoff)
                .steps
                .iter()
                .all(|s| matches!(s, Step::Gpu { .. })));
        }
        let mut strict = c.clone();
        strict.gpu.min_bank_rows = 8192;
        let p = plan(&FLASH_NEXT, &c, FamilyMask::ALL.into(), 16343);
        assert!(executable_plan(&FLASH_NEXT, &strict, &p, 16343, Some(0))
            .steps
            .iter()
            .all(|s| matches!(s, Step::Gpu { .. })));
        let rebank = Plan {
            steps: vec![
                Step::Pipe {
                    a: 0..5000,
                    b: 5000..10000,
                },
                Step::Pipe {
                    a: 10000..15000,
                    b: 15000..20000,
                },
            ],
            t_gpu_ms: 1.0,
            t_pipe_ms: 0.5,
        };
        assert!(executable_plan(&FLASH_NEXT, &c, &rebank, 20000, Some(0))
            .steps
            .iter()
            .all(|s| matches!(s, Step::Gpu { .. })));
        // Only the rebalanced geometry is new: unaligned, split, too-small or
        // not-ending-at-the-request pairs stay GPU-only.
        let admitted = |a: std::ops::Range<usize>, b: std::ops::Range<usize>, rows: usize| {
            let p = Plan {
                steps: vec![Step::Pipe { a, b }],
                t_gpu_ms: 1.0,
                t_pipe_ms: 0.5,
            };
            executable_plan(&FLASH_NEXT, &c, &p, rows, Some(0))
                .steps
                .iter()
                .any(|s| matches!(s, Step::Pipe { .. }))
        };
        assert!(admitted(0..4672, 4672..9319, 9319));
        assert!(admitted(49152..53376, 53376..57556, 57556));
        assert!(!admitted(0..4650, 4650..9319, 9319), "split not 64-aligned");
        assert!(!admitted(0..4096, 4096..8192, 8192), "single-chunk split");
        assert!(
            !admitted(0..1024, 1024..9319, 9319),
            "bank A below the minimum, bank B above a chunk"
        );
        assert!(
            !admitted(0..4672, 4672..9000, 9319),
            "does not end the request"
        );
        assert!(!admitted(0..64, 64..9319, 9319), "bank below the minimum");
        assert!(!admitted(4096..8000, 8000..12288, 12288), "unaligned start");
    }
}

/// Names of the Python model (`qkv zab q kvidx gdn.out qsa.o`) as the planner's moved set.
fn moved_names(names: &Value) -> Moved {
    let (mut inproj, mut out) = ([false; 4], [false; 2]);
    for name in names.as_array().unwrap() {
        match name.as_str().unwrap() {
            "qkv" => inproj[0] = true,
            "zab" => inproj[1] = true,
            "q" => inproj[2] = true,
            "kvidx" => inproj[3] = true,
            "gdn.out" => out[0] = true,
            "qsa.o" => out[1] = true,
            other => panic!("unknown golden family {other}"),
        }
    }
    Moved::new(inproj, out)
}
fn v5_golden() -> (Calibration, Value) {
    let vectors: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/npu_trunk/golden-v5.json"
    )))
    .unwrap();
    let v = vectors[0].clone();
    let cal: Calibration = serde_json::from_value(v["calibration"].clone()).unwrap();
    cal.validate(&FLASH_NEXT).unwrap();
    (cal, v)
}
#[test]
fn npu_trunk_v5_goldens_match_python() {
    let (cal, v) = v5_golden();
    let (selected, report) = select_families(&FLASH_NEXT, &cal);
    assert_eq!(FamilyMask::from(selected), grouped_mask(&v["selected"]));
    assert_eq!(report.subsets.len(), 15);
    for (actual, expected) in report.subsets.iter().zip(v["subsets"].as_array().unwrap()) {
        assert_eq!(
            FamilyMask::from(actual.families),
            grouped_mask(&expected["families"])
        );
        assert_eq!(actual.selectable, expected["selectable"].as_bool().unwrap());
        assert_eq!(
            actual.beats_margin_16384,
            expected["beats_margin_16384"].as_bool().unwrap()
        );
        close(actual.mean_relative, &expected["mean_relative"]);
        for i in 0..3 {
            close(actual.t_pipe_ms[i], &expected["t_pipe_ms"][i]);
            close(actual.t_gpu_ms[i], &expected["t_gpu_ms"][i]);
        }
    }
    // The Python table also holds the out-family subsets; none of them is selectable and none can win.
    let python_rows = v["subsets"].as_array().unwrap();
    assert_eq!(python_rows.len(), 63);
    for row in python_rows {
        let has_out = row["families"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n == "gdn.out" || n == "qsa.o");
        assert_eq!(row["selectable"].as_bool().unwrap(), !has_out);
    }
    assert!(!cal.families["qsa.o"].selectable && !cal.families["gdn.out"].selectable);
    for expected in v["bank_segments"].as_array().unwrap() {
        let actual = bank_segments_moved(
            &FLASH_NEXT,
            &cal,
            moved_names(&expected["families"]),
            expected["rows"].as_u64().unwrap() as usize,
            expected["prefix"].as_u64().unwrap() as usize,
        );
        for i in 0..2 {
            close(actual.gpu[i], &expected["gpu"][i]);
            close(actual.npu[i], &expected["npu"][i]);
            close(actual.npu_out[i], &expected["npu_out"][i]);
            close(actual.bytes[i], &expected["bytes"][i]);
        }
    }
    for expected in v["pair_times"].as_array().unwrap() {
        let b: Vec<usize> = expected["banks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let actual = pair_time_moved(
            &FLASH_NEXT,
            &cal,
            moved_names(&expected["families"]),
            b[0],
            b[1],
            b[2],
            b[3],
        );
        for (value, key) in [
            (actual.t, "t"),
            (actual.stall, "stall"),
            (actual.extra, "extra"),
            (actual.fill, "fill"),
            (actual.gpu_only, "gpu_only"),
        ] {
            close(value, &expected[key]);
        }
    }
    for family_plans in v["plans"].as_array().unwrap() {
        let moved = moved_names(&family_plans["families"]);
        for expected in family_plans["plans"].as_array().unwrap() {
            let actual = plan_moved(
                &FLASH_NEXT,
                &cal,
                moved,
                expected["rows"].as_u64().unwrap() as usize,
            );
            let range =
                |r: &Value| r[0].as_u64().unwrap() as usize..r[1].as_u64().unwrap() as usize;
            let steps: Vec<_> = expected["steps"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| match s.get("gpu") {
                    Some(r) => Step::Gpu { chunk: range(r) },
                    None => Step::Pipe {
                        a: range(&s["pipe"][0]),
                        b: range(&s["pipe"][1]),
                    },
                })
                .collect();
            assert_eq!(
                actual.steps, steps,
                "{} {}",
                family_plans["families"], expected["python_plan"]
            );
            close(actual.t_pipe_ms, &expected["t_pipe_ms"]);
            close(actual.t_gpu_ms, &expected["t_gpu_ms"]);
        }
    }
}
#[test]
fn npu_trunk_v5_model_terms() {
    let (cal, _) = v5_golden();
    let all = Moved::of(&FLASH_NEXT, FamilyMask::ALL.into());
    // Stall = pipeline fill only: no steady-state stall, fill is the first in-proj chain, contention is byte-linear.
    let pair = pair_time_moved(&FLASH_NEXT, &cal, all, 8192, 0, 8151, 8192);
    assert_eq!(pair.stall, 0.0);
    assert_eq!(
        pair.fill,
        bank_segments_moved(&FLASH_NEXT, &cal, all, 8192, 0).npu[0]
    );
    assert!(
        (pair.extra - 200.0).abs() < 0.5,
        "16K in-pipeline contention {}",
        pair.extra
    );
    // The out families never enter the in-proj mask, so the load-time selection cannot contain them.
    assert_eq!(
        Moved::of(&FLASH_NEXT, FamilyMask::ALL.into()).out,
        [false, false]
    );
    // A QSA out family stalls on the order C_A C_B P_A P_B: out chain only (no in-proj) stalls nothing here, with the
    // in-proj chains the post-GEMM window shrinks and the layer stalls.
    let qsa_out = pair_time_moved(
        &FLASH_NEXT,
        &cal,
        Moved::new([false; 4], [false, true]),
        8192,
        0,
        8151,
        8192,
    );
    assert_eq!(qsa_out.stall, 0.0);
    let both = pair_time_moved(
        &FLASH_NEXT,
        &cal,
        Moved::new([true, true, true, true], [false, true]),
        8192,
        0,
        8151,
        8192,
    );
    assert!(both.stall > 0.0);
    // The out chain is NPU time and DRAM bytes, and the GPU o-proj GEMM leaves the GPU segment.
    let seg = bank_segments_moved(
        &FLASH_NEXT,
        &cal,
        Moved::new([false; 4], [true, true]),
        8192,
        0,
    );
    let base = bank_segments_moved(
        &FLASH_NEXT,
        &cal,
        Moved::new([false; 4], [false; 2]),
        8192,
        0,
    );
    assert!(seg.npu_out[0] > 0.0 && seg.npu_out[1] > 0.0 && seg.npu == base.npu);
    assert!(seg.bytes[1] > 0.0 && seg.gpu[1] < base.gpu[1]);
    // Selection stays in-proj: gating an in-proj family out of selection removes its subsets from the contest.
    let mut gated = cal.clone();
    gated.families.get_mut("q").unwrap().selectable = false;
    let (picked, report) = select_families(&FLASH_NEXT, &gated);
    assert!(
        report.subsets.iter().any(|r| !r.selectable)
            && !FamilyMask::from(picked).contains(Family::QsaQ)
    );
}
#[test]
fn npu_trunk_v5_validation() {
    let (cal, _) = v5_golden();
    let mut v = serde_json::to_value(&cal).unwrap();
    v["families"]["qsa.o"]
        .as_object_mut()
        .unwrap()
        .remove("out");
    assert!(serde_json::from_value::<Calibration>(v)
        .unwrap()
        .validate(&FLASH_NEXT)
        .is_err());
    let mut v = serde_json::to_value(&cal).unwrap();
    v.as_object_mut().unwrap().remove("qsa_order");
    assert!(serde_json::from_value::<Calibration>(v)
        .unwrap()
        .validate(&FLASH_NEXT)
        .is_err());
    let mut v = serde_json::to_value(&cal).unwrap();
    v["npu"].as_object_mut().unwrap().remove("tops_alone");
    assert!(serde_json::from_value::<Calibration>(v)
        .unwrap()
        .validate(&FLASH_NEXT)
        .is_err());
    let mut v = serde_json::to_value(&cal).unwrap();
    v["families"]["qkv"]["out"] = v["families"]["qsa.o"]["out"].clone();
    assert!(serde_json::from_value::<Calibration>(v)
        .unwrap()
        .validate(&FLASH_NEXT)
        .is_err());
    let mut v = serde_json::to_value(&cal).unwrap();
    v["families"]["other"] = v["families"]["qkv"].clone();
    assert!(serde_json::from_value::<Calibration>(v)
        .unwrap()
        .validate(&FLASH_NEXT)
        .is_err());
    let mut invalid = cal.clone();
    invalid.contention.ms_per_gb = f64::NAN;
    assert!(invalid.validate(&FLASH_NEXT).is_err());
    // v4 files (no out families, no per-byte contention) still load unchanged.
    let v4: Calibration =
        serde_json::from_str(include_str!("calibrations/halo-xdna2-overlay-v4.json")).unwrap();
    v4.validate(&FLASH_NEXT).unwrap();
    assert!(v4.qsa_order.is_none() && v4.contention.ms_per_gb == 0.0 && v4.families.len() == 4);
    let roundtrip: Value = serde_json::to_value(&v4).unwrap();
    assert!(
        roundtrip["contention"].get("ms_per_gb").is_none() && roundtrip.get("qsa_order").is_none()
    );
}
