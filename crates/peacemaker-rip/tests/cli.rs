//! The `peacemaker-rip` command line: `check`, `build` (and the legacy positional form),
//! `fmt`, `explain`, `help`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn rip(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_peacemaker-rip"))
        .args(args)
        .env("ROCR_VISIBLE_DEVICES", "-1")
        .env("HIP_VISIBLE_DEVICES", "-1")
        .output()
        .expect("run peacemaker-rip")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rip-cli-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn write(dir: &Path, name: &str, body: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, body).expect("write fixture");
    p.display().to_string()
}

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

fn good() -> String {
    kernel(r#"op(wg.isa(), "v_mov_b32_e32 v0, 0", &[v(0)], &[])?;"#)
}

/// A store handed over a barrier without its wait: the driver refuses it (R0301).
fn broken() -> String {
    kernel(
        r#"let t = wg.lds::<T>("t", 0, 16)?;
    let (t, st) = wg.ds_store(t, Instruction::new("ds_store_b32 v0, v1", vec![], vec![v(0), v(1)]).memory(MemoryClass::DsStore))?;
    let (t,) = wg.barrier((ready(t, st),))?;
    wg.ds_load(&t, Instruction::new("ds_load_b32 v2, v0", vec![v(2)], vec![v(0)]).memory(MemoryClass::DsLoad))?;"#,
    )
}

#[test]
fn legacy_form_writes_the_same_assembly_as_build() {
    let dir = scratch("legacy");
    let src = write(&dir, "k.rip", &good());
    let legacy_out = dir.join("legacy.s").display().to_string();
    let build_out = dir.join("build.s").display().to_string();
    let a = rip(&[&src, "gfx1201", "v", &legacy_out]);
    let b = rip(&["build", &src, "gfx1201", "v", &build_out]);
    assert!(a.status.success(), "{}", text(&a.stderr));
    assert!(b.status.success(), "{}", text(&b.stderr));
    let legacy_s = std::fs::read_to_string(&legacy_out).unwrap();
    assert!(legacy_s.contains("s_endpgm"), "{legacy_s}");
    assert_eq!(legacy_s, std::fs::read_to_string(&build_out).unwrap());
    assert!(
        text(&a.stderr).contains("1 kernel(s) for gfx1201 ->"),
        "{}",
        text(&a.stderr)
    );
    assert_eq!(
        text(&a.stderr).replace(&legacy_out, "OUT"),
        text(&b.stderr).replace(&build_out, "OUT")
    );
}

#[test]
fn legacy_both_writes_one_file_per_arch() {
    let dir = scratch("both");
    let src = write(&dir, "k.rip", &good());
    let out = dir.join("out.s").display().to_string();
    let r = rip(&[&src, "both", "v", &out]);
    assert!(r.status.success(), "{}", text(&r.stderr));
    assert!(dir.join("out.gfx1151.s").exists() && dir.join("out.gfx1201.s").exists());
}

#[test]
fn build_failure_renders_a_diagnostic_and_json_carries_the_code() {
    let dir = scratch("build_fail");
    let src = write(&dir, "bad.rip", &broken());
    let out = dir.join("bad.s").display().to_string();
    let r = rip(&["build", &src, "gfx1151", "v", &out]);
    assert_eq!(r.status.code(), Some(1));
    assert!(
        text(&r.stderr).contains("error[R0301]"),
        "{}",
        text(&r.stderr)
    );
    let j = rip(&["build", "--json", &src, "gfx1151", "v", &out]);
    assert_eq!(j.status.code(), Some(1));
    let doc: serde_json::Value = serde_json::from_slice(&j.stdout).expect("json on stdout");
    assert_eq!(doc["ok"], false);
    assert_eq!(doc["diagnostics"][0]["code"], "R0301");
}

#[test]
fn build_renders_the_scripts_own_error_as_r0502() {
    let dir = scratch("build_r0502");
    let src = write(&dir, "only1151.rip", "fn emit(spec: Spec) -> Result<Emitted, String> {\n    Err(\"only1151: built for gfx1151 only\".to_string())\n}\n");
    let out = dir.join("o.s").display().to_string();
    let r = rip(&["build", &src, "gfx1201", "v", &out]);
    assert_eq!(r.status.code(), Some(1));
    assert!(
        text(&r.stderr).contains("error[R0502]")
            && text(&r.stderr).contains("built for gfx1151 only"),
        "{}",
        text(&r.stderr)
    );
    let m = rip(&[
        "build", "--json", "--module", &src, "gfx1201", "v", "m", &out,
    ]);
    assert_eq!(m.status.code(), Some(1));
    let doc: serde_json::Value = serde_json::from_slice(&m.stdout).expect("json on stdout");
    assert_eq!(doc["diagnostics"][0]["code"], "R0502");
}

#[test]
fn check_on_a_valid_kernel_exits_zero_with_a_summary_line_per_target() {
    let dir = scratch("check_ok");
    let src = write(&dir, "k.rip", &good());
    let r = rip(&["check", &src, "--variant", "v"]);
    let err = text(&r.stderr);
    assert_eq!(r.status.code(), Some(0), "{err}");
    assert!(
        err.contains("ok gfx1151/v: ")
            && err.contains("instructions")
            && err.contains("VGPRs")
            && err.contains("SGPRs")
            && err.contains("LDS"),
        "{err}"
    );
    assert!(err.contains("ok gfx1201/v: "), "{err}");
    let all = rip(&["check", &src, "--variant", "v", "--arch", "all"]);
    assert_eq!(all.status.code(), Some(0));
    assert!(
        text(&all.stderr).contains("ok gfx1100/v: "),
        "{}",
        text(&all.stderr)
    );
}

#[test]
fn check_json_describes_every_target() {
    let dir = scratch("check_json");
    let src = write(&dir, "k.rip", &good());
    let r = rip(&[
        "check",
        &src,
        "--variant",
        "v",
        "--arch",
        "gfx1201",
        "--json",
    ]);
    assert_eq!(r.status.code(), Some(0), "{}", text(&r.stderr));
    let doc: serde_json::Value = serde_json::from_slice(&r.stdout).expect("json on stdout");
    assert_eq!(doc["ok"], true);
    assert_eq!(doc["file"], src.as_str());
    let results = doc["results"].as_array().unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0]["arch"], "gfx1201");
    assert_eq!(results[0]["variant"], "v");
    assert!(results[0]["resources"]["instructions"].as_u64().unwrap() >= 1);
}

#[test]
fn check_on_a_broken_kernel_exits_one_and_json_has_code_and_span() {
    let dir = scratch("check_bad");
    let src = write(&dir, "bad.rip", &broken());
    let r = rip(&["check", &src, "--variant", "v", "--arch", "gfx1151"]);
    let err = text(&r.stderr);
    assert_eq!(r.status.code(), Some(1), "{err}");
    assert!(
        err.contains("error[R0301]") && err.contains("error gfx1151/v: 1 error"),
        "{err}"
    );
    let j = rip(&[
        "check",
        &src,
        "--variant",
        "v",
        "--arch",
        "gfx1151",
        "--json",
    ]);
    assert_eq!(j.status.code(), Some(1));
    let doc: serde_json::Value = serde_json::from_slice(&j.stdout).expect("json on stdout");
    assert_eq!(doc["ok"], false);
    let result = &doc["results"][0];
    assert_eq!(result["ok"], false);
    assert!(result["resources"].is_null());
    let d = &result["diagnostics"][0];
    assert_eq!(d["code"], "R0301");
    assert_eq!(d["severity"], "error");
    assert!(
        d["span"]["line"].as_u64().unwrap() >= 1 && d["span"]["col"].as_u64().unwrap() >= 1,
        "{d}"
    );
}

#[test]
fn check_without_declared_variants_is_a_usage_error() {
    let dir = scratch("check_usage");
    let src = write(&dir, "k.rip", &good());
    let r = rip(&["check", &src]);
    assert_eq!(r.status.code(), Some(2));
    assert!(text(&r.stderr).contains("--variant"), "{}", text(&r.stderr));
    assert_eq!(rip(&["check"]).status.code(), Some(2));
    assert_eq!(rip(&["check", &src, "--bogus"]).status.code(), Some(2));
}

#[test]
fn check_discovers_const_variants_and_arches_with_no_flags() {
    // `good()` declares neither const: the usage error still names VARIANTS.
    let dir = scratch("check_discover");
    let src = write(&dir, "k.rip", &good());
    let r = rip(&["check", &src]);
    assert_eq!(r.status.code(), Some(2), "{}", text(&r.stderr));
    assert!(text(&r.stderr).contains("VARIANTS"), "{}", text(&r.stderr));
    // A `const VARIANTS` script checks without `--variant`.
    let with_variants = format!("pub const VARIANTS: &[&str] = &[\"v\"];\n{}", good());
    let src = write(&dir, "c.rip", &with_variants);
    let r = rip(&["check", &src]);
    assert_eq!(r.status.code(), Some(0), "{}", text(&r.stderr));
    assert!(
        text(&r.stderr).contains("ok gfx1151/v: ") && text(&r.stderr).contains("ok gfx1201/v: "),
        "{}",
        text(&r.stderr)
    );
    // A `const ARCHES` script narrows the default arch list ...
    let with_arches = format!(
        "pub const VARIANTS: &[&str] = &[\"v\"];\npub const ARCHES: &[&str] = &[\"gfx1151\"];\n{}",
        good()
    );
    let src = write(&dir, "a.rip", &with_arches);
    let r = rip(&["check", &src]);
    let err = text(&r.stderr);
    assert_eq!(r.status.code(), Some(0), "{err}");
    assert!(
        err.contains("ok gfx1151/v: ") && !err.contains("gfx1201"),
        "{err}"
    );
    // ... and `--arch all` means the declared ARCHES, not all three arches.
    let r = rip(&["check", &src, "--arch", "all"]);
    let err = text(&r.stderr);
    assert_eq!(r.status.code(), Some(0), "{err}");
    assert!(
        err.contains("ok gfx1151/v: ") && !err.contains("gfx1201"),
        "{err}"
    );
}

#[test]
fn check_flags_beat_the_consts_on_both_axes() {
    let dir = scratch("check_precedence");
    let body = format!(
        "pub const VARIANTS: &[&str] = &[\"v\"];\npub const ARCHES: &[&str] = &[\"gfx1151\"];\n{}",
        good()
    );
    let src = write(&dir, "p.rip", &body);
    // `--variant` beats `const VARIANTS`: the requested name is compiled,
    // not the declared one (the fixture accepts any variant string).
    let r = rip(&["check", &src, "--variant", "other", "--arch", "gfx1151"]);
    let err = text(&r.stderr);
    assert_eq!(r.status.code(), Some(0), "{err}");
    assert!(err.contains("ok gfx1151/other: "), "{err}");
    // `--arch` beats `const ARCHES`: the second arch is compiled too.
    let r = rip(&["check", &src, "--arch", "gfx1151,gfx1201"]);
    let err = text(&r.stderr);
    assert_eq!(r.status.code(), Some(0), "{err}");
    assert!(err.contains("ok gfx1201/v: "), "{err}");
}

#[test]
fn check_empty_variants_and_bogus_arch_are_usage_errors() {
    let dir = scratch("check_bad_lists");
    let empty = format!("pub const VARIANTS: &[&str] = &[];\n{}", good());
    let src = write(&dir, "e.rip", &empty);
    let r = rip(&["check", &src]);
    assert_eq!(r.status.code(), Some(2), "{}", text(&r.stderr));
    assert!(text(&r.stderr).contains("VARIANTS"), "{}", text(&r.stderr));
    // A bogus arch name from the flag is a usage error.
    let with_variants = format!("pub const VARIANTS: &[&str] = &[\"v\"];\n{}", good());
    let src = write(&dir, "b.rip", &with_variants);
    let r = rip(&["check", &src, "--arch", "gfx9999"]);
    assert_eq!(r.status.code(), Some(2), "{}", text(&r.stderr));
    // ... and so is one from the script's own `const ARCHES`.
    let bad_arches = format!(
        "pub const VARIANTS: &[&str] = &[\"v\"];\npub const ARCHES: &[&str] = &[\"gfx9999\"];\n{}",
        good()
    );
    let src = write(&dir, "ba.rip", &bad_arches);
    let r = rip(&["check", &src]);
    assert_eq!(r.status.code(), Some(2), "{}", text(&r.stderr));
}

#[test]
fn check_refusal_under_an_explicit_arch_is_an_error() {
    // `good()` builds on every arch; a gfx1151-only guard refuses gfx1201
    // with the script's own `Err`, which stays exit 1 under the flag.
    let dir = scratch("check_refuse");
    let guarded = format!(
        "pub const VARIANTS: &[&str] = &[\"v\"];\nfn guard(spec: Spec) -> Result<(), String> {{ if spec.arch != Arch::Gfx1151 {{ return Err(\"only gfx1151\".into()) }} Ok(()) }}\n{}",
        good().replace(
            "let mut p = RegPlan::new(256, 104)?;",
            "guard(spec)?;\n    let mut p = RegPlan::new(256, 104)?;"
        )
    );
    let src = write(&dir, "g.rip", &guarded);
    let r = rip(&["check", &src, "--arch", "gfx1201"]);
    assert_eq!(r.status.code(), Some(1), "{}", text(&r.stderr));
    assert!(
        text(&r.stderr).contains("only gfx1151"),
        "{}",
        text(&r.stderr)
    );
}

#[test]
fn explain_prints_a_section_lists_codes_and_rejects_unknown_ones() {
    let r = rip(&["explain", "R0301"]);
    assert_eq!(r.status.code(), Some(0));
    let out = text(&r.stdout);
    assert!(out.contains("R0301") && out.contains("barrier"), "{out}");
    let list = rip(&["explain", "--list"]);
    assert_eq!(list.status.code(), Some(0));
    assert!(
        text(&list.stdout).lines().any(|l| l.starts_with("R0301  ")),
        "{}",
        text(&list.stdout)
    );
    let bad = rip(&["explain", "R9999"]);
    assert_eq!(bad.status.code(), Some(1));
    assert!(
        text(&bad.stderr).contains("--list"),
        "{}",
        text(&bad.stderr)
    );
}

#[test]
fn fmt_check_reports_messy_files_and_fmt_rewrites_them() {
    let dir = scratch("fmt");
    let messy = write(&dir, "messy.rip", "fn   helper( x:u32 )->u32{x+1}\n");
    let r = rip(&["fmt", "--check", &messy]);
    assert_eq!(r.status.code(), Some(1), "{}", text(&r.stderr));
    assert!(text(&r.stdout).contains(&messy), "{}", text(&r.stdout));
    let w = rip(&["fmt", &messy]);
    assert_eq!(w.status.code(), Some(0), "{}", text(&w.stderr));
    let formatted = std::fs::read_to_string(&messy).unwrap();
    assert_ne!(formatted, "fn   helper( x:u32 )->u32{x+1}\n");
    assert_eq!(rip(&["fmt", "--check", &messy]).status.code(), Some(0));
}

#[test]
fn help_and_no_arguments() {
    let h = rip(&["help"]);
    assert_eq!(h.status.code(), Some(0));
    for word in ["check", "build", "fmt", "explain"] {
        assert!(text(&h.stdout).contains(word), "{}", text(&h.stdout));
    }
    assert_eq!(rip(&["--help"]).status.code(), Some(0));
    assert_eq!(rip(&[]).status.code(), Some(2));
}
