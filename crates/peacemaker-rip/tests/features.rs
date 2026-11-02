//! `features::collect` integration tests: stable JSON keys, hash shapes,
//! resource parity with `check`, sorted emulator coverage, route-dependent
//! cost fields, and structured per-target failures. Everything runs offline;
//! no test touches a GPU.

use peacemaker_isa::Arch;
use peacemaker_rip::diag::Severity;
use serde_json::Value;

const GATHER: &str = include_str!("../../peacemaker-isa/src/kernels/qsa_gather.rip");

/// A kernel script whose `emit` runs `body` inside a `Workgroup` scope.
fn kernel(body: &str) -> String {
    format!(
        r#"region T;
fn emit(spec: Spec) -> Result<Emitted, String> {{
    let kspec = KernelSpec {{
        kernel_id: "t".into(), variant: "v".into(), arch: spec.arch, symbol: "t_kernel".into(),
        kernargs: KernargLayout::new(0), user_sgpr_count: 2, system_sgpr_workgroup_id_y: false,
        workgroup_size: 32, group_segment_fixed_size: 1024, wave32: true, cu_mode: false,
    }};
    let mut p = RegPlan::new(256, 104)?;
    for i in 0..4u8 {{ p.v::<8>("vec", 8 * i, Live::Whole)?; }}
    p.s::<8>("scalars", 0, Live::Whole)?;
    let mut b = Builder::new(kspec, p);
    let mut wg = Workgroup::new(&mut b, spec.arch)?;
    {body}
    let end = wg.exit(".Lend")?;
    wg.end(end)?;
    b.finish()
}}
"#
    )
}

/// A loop-free kernel: no WMMA, no natural loops in the lifted object.
fn tiny() -> String {
    kernel(r#"op(wg.isa(), "v_mov_b32_e32 v0, 0", &[v(0)], &[])?;"#)
}

/// A kernel `check` rejects: a store handed over a barrier without its wait.
fn broken() -> String {
    kernel(
        r#"let t = wg.lds::<T>("t", 0, 16)?;
    op(wg.isa(), "v_mov_b32_e32 v1, 0", &[v(1)], &[])?;
    let (t, st) = wg.ds_store(t, Instruction::new("ds_store_b32 v0, v1", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;
    let (t,) = wg.barrier((ready(t, st),))?;
    wg.ds_load(&t, Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)]).memory(MemoryClass::DsLoad))?;"#,
    )
}

fn collect(source: &str, arches: &[Arch], variants: &[&str], routes: Option<&str>) -> Value {
    let variants: Vec<String> = variants.iter().map(|s| s.to_string()).collect();
    peacemaker_rip::features::collect(source, "test.rip", arches, &variants, routes)
}

fn target<'a>(report: &'a Value, key: &str) -> &'a Value {
    let t = &report["targets"][key];
    assert!(t.is_object(), "missing target {key}: {}", report["targets"]);
    t
}

fn keys(v: &Value) -> Vec<String> {
    let mut out: Vec<String> = v.as_object().expect("object").keys().cloned().collect();
    out.sort_unstable();
    out
}

/// Toolchain-dependent facts (linked build, lift, emulator, cost) need the
/// `certify` feature; without it every target carries a `toolchain` error.
fn require_toolchain(t: &Value) -> bool {
    if t["error"]["stage"] == "toolchain" {
        eprintln!("note: certify feature off; skipping toolchain-dependent assertions");
        return false;
    }
    true
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

#[test]
fn report_roots_have_stable_keys_and_hashes() {
    let report = collect(GATHER, &[Arch::Gfx1151], &["Attend"], None);
    assert_eq!(
        keys(&report),
        [
            "error",
            "file",
            "ok",
            "routes_supplied",
            "schema",
            "source_sha256",
            "targets"
        ]
    );
    assert_eq!(report["schema"], 1);
    assert_eq!(report["file"], "test.rip");
    assert_eq!(
        report["source_sha256"].as_str().expect("source hash"),
        peacemaker_rip::features::source_sha256(GATHER)
    );
    // Resources and the assembly hash come from the emitted text, before the
    // toolchain gate, so they are present with or without `certify`.
    assert_eq!(report["ok"], target(&report, "gfx1151/Attend")["ok"]);
    assert!(report["error"].is_null(), "{}", report["error"]);

    let t = target(&report, "gfx1151/Attend");
    assert_eq!(
        keys(t),
        [
            "arch",
            "assembly_sha256",
            "cost",
            "emulator",
            "error",
            "ok",
            "resources",
            "variant"
        ]
    );
    assert_eq!(t["arch"], "gfx1151");
    assert_eq!(t["variant"], "Attend");
    let hash = t["assembly_sha256"].as_str().expect("assembly hash");
    assert!(is_hex64(hash), "{hash}");
    if t["ok"].as_bool() != Some(true) {
        // Without the `certify` feature every target carries a `toolchain`
        // error; that is the only acceptable non-ok outcome here.
        assert_eq!(t["error"]["stage"], "toolchain", "{}", t["error"]);
        assert_eq!(report["ok"], false);
        return;
    }
    assert!(t["error"].is_null(), "{}", t["error"]);
}

#[test]
fn resources_match_the_typed_check_resources() {
    let report = collect(GATHER, &[Arch::Gfx1151], &["Attend"], None);
    let t = target(&report, "gfx1151/Attend");
    assert_eq!(
        keys(&t["resources"]),
        ["instructions", "lds_bytes", "sgprs", "vgprs", "wave_size"]
    );
    let checked = peacemaker_rip::check(GATHER, "qsa_gather.rip", Arch::Gfx1151, "Attend");
    assert!(
        checked
            .diagnostics
            .iter()
            .all(|d| d.severity != Severity::Error),
        "{:?}",
        checked.diagnostics
    );
    let r = checked.resources.as_ref().expect("typed resources");
    assert_eq!(
        t["resources"]["vgprs"].as_u64().unwrap(),
        u64::from(r.vgprs)
    );
    assert_eq!(
        t["resources"]["sgprs"].as_u64().unwrap(),
        u64::from(r.sgprs)
    );
    assert_eq!(
        t["resources"]["lds_bytes"].as_u64().unwrap(),
        u64::from(r.lds_bytes)
    );
    assert_eq!(
        t["resources"]["instructions"].as_u64().unwrap(),
        r.instructions as u64
    );
    // `wave_size` comes from the emitted descriptor, never the typed resources.
    assert_eq!(t["resources"]["wave_size"].as_u64().unwrap(), 32);
}

#[test]
fn assembly_hash_is_stable_and_variant_sensitive() {
    let a = collect(GATHER, &[Arch::Gfx1151], &["Attend"], None);
    let b = collect(GATHER, &[Arch::Gfx1151], &["Attend"], None);
    assert_eq!(
        a["targets"]["gfx1151/Attend"]["assembly_sha256"],
        b["targets"]["gfx1151/Attend"]["assembly_sha256"]
    );

    let both = collect(GATHER, &[Arch::Gfx1151], &["Attend", "Convert"], None);
    let attend = both["targets"]["gfx1151/Attend"]["assembly_sha256"]
        .as_str()
        .unwrap();
    let convert = both["targets"]["gfx1151/Convert"]["assembly_sha256"]
        .as_str()
        .unwrap();
    assert!(is_hex64(attend) && is_hex64(convert), "{attend} {convert}");
    assert_ne!(attend, convert, "different variants must hash differently");
}

#[test]
fn emulator_counts_are_internally_consistent_and_sorted() {
    let report = collect(GATHER, &[Arch::Gfx1151], &["Attend"], None);
    let t = target(&report, "gfx1151/Attend");
    if !require_toolchain(t) {
        return;
    }
    assert_eq!(
        keys(&t["emulator"]),
        ["instructions", "modeled", "unmodeled"]
    );
    let instructions = t["emulator"]["instructions"]
        .as_u64()
        .expect("instructions");
    let modeled = t["emulator"]["modeled"].as_u64().expect("modeled");
    assert!(instructions > 0, "{instructions}");
    let entries = t["emulator"]["unmodeled"]
        .as_array()
        .expect("unmodeled array");
    let mut sum = modeled;
    let mut names = Vec::new();
    for entry in entries {
        assert_eq!(keys(entry), ["count", "mnemonic"]);
        let name = entry["mnemonic"].as_str().expect("mnemonic");
        assert!(!name.is_empty(), "{entry}");
        let count = entry["count"].as_u64().expect("count");
        assert!(count > 0, "{entry}");
        sum += count;
        names.push(name.to_string());
    }
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        names, sorted,
        "unmodeled entries must be sorted by mnemonic"
    );
    assert_eq!(
        sum, instructions,
        "modeled + unmodeled must cover every instruction"
    );
    // The lifted object is byte-exact, so coverage spans the emitted body.
    assert_eq!(
        sum,
        t["resources"]["instructions"].as_u64().unwrap(),
        "coverage must span the resource instruction count"
    );
}

#[test]
fn cost_loops_retain_route_dependent_null_fields() {
    let report = collect(GATHER, &[Arch::Gfx1151], &["Attend"], None);
    assert_eq!(report["routes_supplied"], false);
    let t = target(&report, "gfx1151/Attend");
    if !require_toolchain(t) {
        return;
    }
    let cost = &t["cost"];
    // `cost.arch` is the `Arch` Debug name, not the `arch.name()` target string.
    assert_eq!(cost["arch"], "Gfx1151");
    // Paired-object only: always null on a single target.
    assert!(
        cost["drain_pressure"].is_null(),
        "{}",
        cost["drain_pressure"]
    );
    let kernels = cost["kernels"].as_array().expect("kernels");
    assert!(!kernels.is_empty(), "{cost}");
    let mut saw_wmma = false;
    for kernel in kernels {
        assert_eq!(keys(kernel), ["loops", "symbol"]);
        for cost_loop in kernel["loops"].as_array().expect("loops") {
            assert_eq!(
                keys(cost_loop),
                [
                    "blocks",
                    "findings",
                    "independent_chains",
                    "k128_epochs",
                    "padding_fraction",
                    "prefetch_distance_epochs",
                    "valu_per_wmma",
                    "vmem_bytes_per_wmma",
                    "vmem_per_wmma",
                    "wait_depth_before_wmma",
                    "wmma_per_k128",
                    "wmma_per_trip",
                ]
            );
            if cost_loop["wmma_per_trip"].as_u64().unwrap() > 0 {
                saw_wmma = true;
                for field in [
                    "wmma_per_trip",
                    "valu_per_wmma",
                    "independent_chains",
                    "wait_depth_before_wmma",
                    "vmem_per_wmma",
                    "vmem_bytes_per_wmma",
                ] {
                    assert!(cost_loop[field].is_number(), "{field}: {cost_loop}");
                }
                for field in [
                    "k128_epochs",
                    "wmma_per_k128",
                    "padding_fraction",
                    "prefetch_distance_epochs",
                ] {
                    assert!(cost_loop[field].is_null(), "{field}: {cost_loop}");
                }
            }
        }
    }
    assert!(saw_wmma, "the Attend WMMA kernel must lift a WMMA loop");
}

#[test]
fn valid_routes_populate_route_dependent_fields() {
    let emitted = peacemaker_rip::compile(GATHER, Arch::Gfx1151, "Attend").expect("compile Attend");
    let symbol = peacemaker_rip::symbol_of(&emitted).expect("symbol");
    assert!(symbol.contains("wmma"), "{symbol}");

    let plain = collect(GATHER, &[Arch::Gfx1151], &["Attend"], None);
    let plain_target = target(&plain, "gfx1151/Attend");
    if !require_toolchain(plain_target) {
        return;
    }
    let plain_loop = plain_target["cost"]["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|k| k["loops"].as_array().unwrap().clone())
        .find(|l| l["wmma_per_trip"].as_u64().unwrap() > 0)
        .expect("a WMMA loop without routes");
    let trips = plain_loop["wmma_per_trip"].as_u64().unwrap();

    let routes = format!(r#"{{"{symbol}": {{"tile_rows": 16, "rows": [5], "k128_epochs": 1}}}}"#);
    let routed = collect(GATHER, &[Arch::Gfx1151], &["Attend"], Some(&routes));
    assert_eq!(routed["routes_supplied"], true);
    let t = target(&routed, "gfx1151/Attend");
    assert_eq!(t["ok"], true, "{}", t["error"]);
    let routed_loop = t["cost"]["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|k| k["loops"].as_array().unwrap().clone())
        .find(|l| l["wmma_per_trip"].as_u64().unwrap() == trips)
        .expect("the same WMMA loop with routes");
    assert_eq!(routed_loop["k128_epochs"], 1);
    assert_eq!(routed_loop["wmma_per_k128"].as_f64().unwrap(), trips as f64);
    let padding = routed_loop["padding_fraction"].as_f64().unwrap();
    assert!((padding - 11.0 / 16.0).abs() < 1e-12, "{padding}");
    // Unrelated facts are unchanged by the routes document; the analysis-only
    // wait depth is recomputed but never asserts anything about routes.
    assert_eq!(t["resources"], plain_target["resources"]);
    assert_eq!(t["assembly_sha256"], plain_target["assembly_sha256"]);
    assert_eq!(t["emulator"], plain_target["emulator"]);
}

#[test]
fn invalid_routes_json_fails_every_target() {
    let src = tiny();
    let report = collect(
        &src,
        &[Arch::Gfx1151, Arch::Gfx1201],
        &["v"],
        Some("{not json"),
    );
    if !require_toolchain(target(&report, "gfx1151/v")) {
        return;
    }
    assert_eq!(report["routes_supplied"], true);
    assert_eq!(report["ok"], false);
    for key in ["gfx1151/v", "gfx1201/v"] {
        let t = target(&report, key);
        assert_eq!(t["ok"], false);
        assert_eq!(t["error"]["stage"], "routes", "{}", t["error"]);
        assert!(!t["error"]["message"].as_str().unwrap().is_empty());
        // The failure is per-target but preserves the check-time facts.
        assert!(!t["resources"].is_null(), "{}", t["resources"]);
        assert!(
            is_hex64(t["assembly_sha256"].as_str().unwrap()),
            "{}",
            t["assembly_sha256"]
        );
    }
}

#[test]
fn semantically_invalid_routes_fail_without_matching_loops() {
    // The tiny kernel lifts no WMMA loop and no named symbol; route semantics
    // must still fail every target.
    let src = tiny();
    let cases = [
        r#"{"k": {"rows": [8], "tile_rows": 0}}"#,
        r#"{"k": {"k128_epochs": 0}}"#,
        r#"{"k": {"k128_epochs": 2, "epoch_starts": {"3": [10]}}}"#,
        r#"{"k": {"epoch_starts": {"3": [10]}}}"#,
    ];
    for routes in cases {
        let report = collect(&src, &[Arch::Gfx1151], &["v"], Some(routes));
        if !require_toolchain(target(&report, "gfx1151/v")) {
            return;
        }
        assert_eq!(report["routes_supplied"], true, "{routes}");
        let t = target(&report, "gfx1151/v");
        assert_eq!(t["error"]["stage"], "routes", "{routes}: {}", t["error"]);
        assert!(
            !t["error"]["message"].as_str().unwrap().is_empty(),
            "{routes}"
        );
        assert!(!t["resources"].is_null(), "{routes}");
        assert!(is_hex64(t["assembly_sha256"].as_str().unwrap()), "{routes}");
    }
}

#[test]
fn compile_failure_is_a_structured_check_error() {
    let src = broken();
    let report = collect(&src, &[Arch::Gfx1151], &["v"], None);
    assert_eq!(report["ok"], false);
    assert!(report["error"].is_null(), "{}", report["error"]);
    let t = target(&report, "gfx1151/v");
    assert_eq!(t["ok"], false);
    assert_eq!(t["error"]["stage"], "check");
    assert!(
        !t["error"]["message"].as_str().unwrap().is_empty(),
        "{}",
        t["error"]
    );
    for field in ["resources", "assembly_sha256", "cost", "emulator"] {
        assert!(t[field].is_null(), "{field}: {}", t[field]);
    }
}

#[test]
fn first_failure_wins_over_later_stages() {
    // A broken source with an invalid routes document still reports `check`:
    // the first failure wins and later stages never overwrite it.
    let src = broken();
    let report = collect(&src, &[Arch::Gfx1151], &["v"], Some("{not json"));
    assert_eq!(report["routes_supplied"], true);
    let t = target(&report, "gfx1151/v");
    assert_eq!(t["error"]["stage"], "check", "{}", t["error"]);
}

#[test]
fn human_features_output_reports_targets_without_a_gpu() {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("rip-features-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("gather.rip");
    std::fs::write(&path, GATHER).expect("write fixture");

    let output = Command::new(env!("CARGO_BIN_EXE_peacemaker-rip"))
        .args([
            "features",
            &path.display().to_string(),
            "--arch",
            "gfx1151",
            "--variant",
            "Attend",
        ])
        .env("ROCR_VISIBLE_DEVICES", "-1")
        .env("HIP_VISIBLE_DEVICES", "-1")
        .output()
        .expect("run peacemaker-rip features");
    assert!(output.status.success() || !output.stdout.is_empty());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("gfx1151/Attend:"), "{stdout}");
    // The human numbers are the same facts as the JSON report: parse them
    // back out and compare against the typed `check` resources below.
    let digits: Vec<u64> = stdout
        .split(|c: char| !c.is_ascii_digit())
        .filter(|w| !w.is_empty())
        .filter_map(|w| w.parse().ok())
        .collect();
    assert!(stdout.contains("instructions,"), "{stdout}");
    assert!(stdout.contains("sha256 source="), "{stdout}");
    assert!(stdout.contains("emulator: modeled="), "{stdout}");

    let json = Command::new(env!("CARGO_BIN_EXE_peacemaker-rip"))
        .args([
            "features",
            &path.display().to_string(),
            "--arch",
            "gfx1151",
            "--variant",
            "Attend",
            "--json",
        ])
        .env("ROCR_VISIBLE_DEVICES", "-1")
        .env("HIP_VISIBLE_DEVICES", "-1")
        .output()
        .expect("run peacemaker-rip features --json");
    let report: Value = serde_json::from_slice(&json.stdout).expect("parse features JSON");
    assert_eq!(report["schema"], 1);
    let rt = &report["targets"]["gfx1151/Attend"];
    for key in ["instructions", "vgprs", "sgprs", "lds_bytes"] {
        let n = rt["resources"][key].as_u64().expect("resource number");
        assert!(digits.contains(&n), "{key}={n} not in {stdout}");
    }
    if rt["ok"].as_bool() == Some(true) {
        assert!(
            json.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&json.stderr)
        );
        assert_eq!(report["ok"], true);
        return;
    }
    assert_eq!(rt["error"]["stage"], "toolchain");
}

#[test]
fn cli_routes_file_populates_fields_and_malformed_routes_fail() {
    use std::process::Command;
    let dir = std::env::temp_dir().join(format!("rip-features-routes-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let src_path = dir.join("gather.rip");
    std::fs::write(&src_path, GATHER).expect("write fixture");
    // The symbol comes from the emitted object at runtime, never hardcoded.
    let emitted = peacemaker_rip::compile(GATHER, Arch::Gfx1151, "Attend").expect("compile Attend");
    let symbol = peacemaker_rip::symbol_of(&emitted).expect("symbol");
    let routes_path = dir.join("routes.json");
    std::fs::write(
        &routes_path,
        format!(r#"{{"{symbol}": {{"tile_rows": 16, "rows": [5], "k128_epochs": 1}}}}"#),
    )
    .expect("write routes");
    let run = |extra: &[String]| {
        Command::new(env!("CARGO_BIN_EXE_peacemaker-rip"))
            .args(
                [
                    "features".to_string(),
                    src_path.display().to_string(),
                    "--arch".to_string(),
                    "gfx1151".to_string(),
                    "--variant".to_string(),
                    "Attend".to_string(),
                    "--json".to_string(),
                ]
                .into_iter()
                .chain(extra.iter().cloned()),
            )
            .env("ROCR_VISIBLE_DEVICES", "-1")
            .env("HIP_VISIBLE_DEVICES", "-1")
            .output()
            .expect("run peacemaker-rip features")
    };
    let out = run(&[format!("--routes={}", routes_path.display())]);
    let routed: Value = serde_json::from_slice(&out.stdout).expect("parse routed features JSON");
    assert_eq!(routed["routes_supplied"], true);
    let t = &routed["targets"]["gfx1151/Attend"];
    if t["error"]["stage"] == "toolchain" {
        eprintln!("note: certify feature off; skipping CLI routes assertions");
        return;
    }
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(t["ok"], true, "{}", t["error"]);
    let cost_loop = t["cost"]["kernels"]
        .as_array()
        .expect("kernels")
        .iter()
        .flat_map(|k| k["loops"].as_array().expect("loops").clone())
        .find(|l| l["wmma_per_trip"].as_u64().unwrap_or(0) > 0)
        .expect("a WMMA loop with routes");
    assert_eq!(cost_loop["k128_epochs"], 1);
    let padding = cost_loop["padding_fraction"].as_f64().expect("padding");
    assert!((padding - 11.0 / 16.0).abs() < 1e-12, "{padding}");

    // The `check` JSON reports the same resource numbers (it has no `wave_size`).
    let check_out = Command::new(env!("CARGO_BIN_EXE_peacemaker-rip"))
        .args([
            "check",
            &src_path.display().to_string(),
            "--arch",
            "gfx1151",
            "--variant",
            "Attend",
            "--json",
        ])
        .env("ROCR_VISIBLE_DEVICES", "-1")
        .env("HIP_VISIBLE_DEVICES", "-1")
        .output()
        .expect("run peacemaker-rip check");
    assert!(
        check_out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&check_out.stderr)
    );
    let check: Value = serde_json::from_slice(&check_out.stdout).expect("parse check JSON");
    let check_res = &check["results"][0]["resources"];
    for key in ["instructions", "vgprs", "sgprs", "lds_bytes"] {
        assert_eq!(t["resources"][key], check_res[key], "{key}");
    }

    std::fs::write(&routes_path, "{not json").expect("overwrite routes");
    let bad = run(&[format!("--routes={}", routes_path.display())]);
    assert_eq!(bad.status.code(), Some(1));
    let failed: Value = serde_json::from_slice(&bad.stdout).expect("parse failed features JSON");
    assert_eq!(failed["routes_supplied"], true);
    assert_eq!(failed["ok"], false);
    let ft = &failed["targets"]["gfx1151/Attend"];
    assert_eq!(ft["ok"], false);
    assert_eq!(ft["error"]["stage"], "routes", "{}", ft["error"]);
}
