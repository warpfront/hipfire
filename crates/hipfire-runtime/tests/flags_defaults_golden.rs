// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flags migration gate GB: the defaults golden (CPU only).
//!
//! For every gfx target `ArchCaps` knows and an EMPTY configuration (the process
//! config resolved from no user, registry, env or CLI layer; the process
//! environment is never read) this test dumps the complete resolved value of
//! every `rdna_compute::FeatureFlags` field, every `ArchCaps` accessor, the
//! `hipfire_runtime::RuntimeConfig` defaults, the schema defaults and the empty
//! process config, as sorted `section<TAB>arch<TAB>key=value` rows under
//! `tests/golden/flags-defaults/`. The test fails on any difference.
//!
//! The call-site defaults of `developer_bool/developer_var/developer_typed_bool`
//! are statically extracted into `developer-callsites.tsv` by
//! `scripts/flags-golden-extract.py`; `scripts/flags-golden.sh` runs both halves.
//!
//! A migration slice may change the golden only by removing or renaming rows its
//! ledger lists, with equal values.
//!
//! Regenerate (no env var; review the diff before committing):
//!
//! ```text
//! scripts/flags-golden.sh --regen
//! # or: cargo test -p hipfire-runtime --test flags_defaults_golden -- --ignored regenerate_flags_defaults_golden
//! ```

use hipfire_config::{resolve, ConfigValue, ProcessConfig, FIELDS};
use hipfire_runtime::config::RuntimeConfig;
use rdna_compute::arch_caps::ArchCaps;
use rdna_compute::feature_flags::{graph_ar_default, FeatureFlags};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The gfx strings `ArchCaps::new` recognises (its twenty `is_gfx*` atoms).
const ARCHES: [&str; 20] = [
    "gfx906", "gfx908", "gfx940", "gfx941", "gfx942", "gfx1010", "gfx1011", "gfx1012", "gfx1030",
    "gfx1031", "gfx1032", "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151",
    "gfx1152", "gfx1200", "gfx1201",
];

const MMQ_BATCHES: [usize; 8] = [1, 8, 32, 64, 128, 256, 512, 4096];

const REGEN_HINT: &str = "scripts/flags-golden.sh --regen";

/// Every non-arg public accessor of `ArchCaps` except `arch` and `should_use_mmq`.
macro_rules! caps_accessors {
    ($($method:ident),* $(,)?) => {
        const CAPS_ACCESSOR_NAMES: &[&str] = &[$(stringify!($method)),*];
        fn caps_rows(caps: &ArchCaps) -> Vec<(&'static str, String)> {
            vec![$((stringify!($method), caps.$method().to_string())),*]
        }
    };
}

caps_accessors!(
    is_gfx906,
    is_gfx908,
    is_gfx1010,
    is_gfx1011,
    is_gfx1012,
    is_gfx1030,
    is_gfx1031,
    is_gfx1032,
    is_gfx1100,
    is_gfx1101,
    is_gfx1102,
    is_gfx1103,
    is_gfx1150,
    is_gfx1151,
    is_gfx1152,
    is_gfx1200,
    is_gfx1201,
    is_gfx940,
    is_gfx941,
    is_gfx942,
    is_gcn5,
    is_cdna1,
    is_rdna1,
    is_rdna1p1,
    is_rdna2,
    is_rdna3,
    is_rdna3_dgpu,
    is_rdna3p5,
    is_rdna3_discrete,
    is_rdna35_apu,
    supports_mq4_lloyd_mb4,
    supports_mq3_lloyd_mb4,
    supports_ds4_f16_compressor_cache,
    has_iu4_wmma,
    supports_dflash_hidden_scatter_fusion,
    supports_dflash_f16_residual_fusions,
    supports_dflash_f16_projection_fusions,
    supports_dflash_gdn_pre_fusions,
    supports_dflash_fa_batch_fusions,
    is_rdna4,
    is_cdna3,
    has_wmma,
    has_wmma_w32,
    has_wmma_w32_gfx12,
    has_gfx11_plus_simt,
    has_qwen4_parallel_qsa_gdn,
    has_dot2_f32_f16,
    has_mmq,
    is_gcn5_wave64,
    is_wave32,
    is_wave64_native,
    has_hfq3_sdot4,
    has_hfq3_dp4a,
    has_hfq3_mmq,
    has_hfq4_mmq,
    has_cdna3_lds_gemv,
    gemv_dp4a_enabled,
    gemv_prefetch_enabled,
    gemv_rows_default,
);

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn golden_dir() -> PathBuf {
    repo_root().join("tests/golden/flags-defaults")
}

/// The process config of an empty configuration: built-in schema defaults only.
fn empty_process_config() -> ProcessConfig {
    ProcessConfig::from_resolved(&resolve(Vec::new()).expect("resolve an empty layer stack"))
        .expect("process config of an empty layer stack")
}

/// Flatten a `{:#?}` struct dump into `(field, value)` rows. Nested values are
/// joined onto one line.
fn debug_fields(dump: &str) -> Vec<(String, String)> {
    let mut rows: Vec<(String, String)> = Vec::new();
    for line in dump.lines().skip(1) {
        if line == "}" {
            break;
        }
        let top = line
            .strip_prefix("    ")
            .filter(|rest| !rest.starts_with(' '));
        let field = top.and_then(|rest| {
            let (name, value) = rest.split_once(": ")?;
            let plain = name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
            plain.then(|| (name.to_owned(), value.to_owned()))
        });
        match field {
            Some(row) => rows.push(row),
            None => {
                let last = rows
                    .last_mut()
                    .expect("continuation before the first field");
                last.1.push_str(line.trim());
            }
        }
    }
    for row in &mut rows {
        row.1 = row
            .1
            .trim_end_matches(',')
            .replace(",)", ")")
            .replace(",]", "]");
    }
    rows
}

fn push_rows(out: &mut Vec<String>, section: &str, arch: &str, rows: Vec<(String, String)>) {
    for (key, value) in rows {
        assert!(
            !value.contains('\t') && !value.contains('\n'),
            "{section} {key}: unrenderable value"
        );
        out.push(format!("{section}\t{arch}\t{key}={value}"));
    }
}

fn file_text(header: &str, mut rows: Vec<String>) -> String {
    rows.sort();
    rows.dedup();
    let mut text = String::new();
    text.push_str("# Flags migration gate GB: defaults golden. Do not edit by hand.\n");
    text.push_str(&format!("# regenerate: {REGEN_HINT}\n"));
    text.push_str(&format!("# {header}\n"));
    text.push_str("# columns: section<TAB>arch<TAB>key=value (sorted)\n");
    for row in rows {
        text.push_str(&row);
        text.push('\n');
    }
    text
}

fn render_files() -> Vec<(&'static str, String)> {
    let config = empty_process_config();

    let mut flags_rows = Vec::new();
    let mut caps_rows_out = Vec::new();
    for arch in ARCHES {
        let flags = FeatureFlags::from_process_config(arch, &config);
        push_rows(
            &mut flags_rows,
            "feature_flags",
            arch,
            debug_fields(&format!("{flags:#?}")),
        );
        flags_rows.push(format!(
            "feature_flags_host\t{arch}\tgraph_ar_default_windows={}",
            graph_ar_default(arch, true)
        ));
        let caps = ArchCaps::new(arch, Arc::new(flags));
        let mut rows: Vec<(String, String)> = caps_rows(&caps)
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect();
        for batch in MMQ_BATCHES {
            rows.push((
                format!("should_use_mmq[{batch}]"),
                caps.should_use_mmq(batch).to_string(),
            ));
        }
        rows.push(("arch".to_owned(), caps.arch().to_owned()));
        push_rows(&mut caps_rows_out, "archcaps", arch, rows);
    }

    let mut runtime_rows = Vec::new();
    let runtime = RuntimeConfig::from_process_config(&config);
    push_rows(
        &mut runtime_rows,
        "runtime_config",
        "*",
        debug_fields(&format!("{runtime:#?}")),
    );

    let mut config_rows = Vec::new();
    for field in FIELDS {
        config_rows.push(format!(
            "schema_default\t*\t{}={:?}",
            field.key, field.default
        ));
    }
    let resolved = resolve(Vec::new()).expect("resolve an empty layer stack");
    for (key, value) in &resolved.values {
        config_rows.push(format!("schema_resolved\t*\t{key}={:?}", value.value));
    }
    for (key, value) in &config.values.values {
        let rendered = match value {
            ConfigValue::Null => "null".to_owned(),
            other => format!("{other:?}"),
        };
        config_rows.push(format!("process_config\t*\t{key}={rendered}"));
    }
    config_rows.push(format!(
        "process_config\t*\tschema_version={}",
        config.schema_version
    ));

    vec![
        (
            "feature-flags.tsv",
            file_text(
                "sections: feature_flags (every FeatureFlags field via Debug, from_process_config over the empty process config, linux host), feature_flags_host",
                flags_rows,
            ),
        ),
        (
            "archcaps.tsv",
            file_text(
                "sections: archcaps (every public accessor, should_use_mmq at fixed batches)",
                caps_rows_out,
            ),
        ),
        (
            "runtime-config.tsv",
            file_text(
                "sections: runtime_config (RuntimeConfig::from_process_config over the empty process config)",
                runtime_rows,
            ),
        ),
        (
            "config-defaults.tsv",
            file_text(
                "sections: schema_default (FIELDS defaults), schema_resolved (resolve of no layers), process_config (the empty ProcessConfig)",
                config_rows,
            ),
        ),
    ]
}

#[test]
fn defaults_match_golden() {
    let mut failures = Vec::new();
    for (name, want_rendered) in render_files() {
        let path = golden_dir().join(name);
        let golden = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}; run {REGEN_HINT}", path.display()));
        if golden != want_rendered {
            failures.push(format!("{name}: {}", first_diff(&golden, &want_rendered)));
        }
    }
    assert!(
        failures.is_empty(),
        "defaults diverged from tests/golden/flags-defaults; if intended run {REGEN_HINT}\n{}",
        failures.join("\n")
    );
}

#[test]
fn arch_list_is_every_arch_caps_atom() {
    let source = std::fs::read_to_string(repo_root().join("crates/rdna-compute/src/arch_caps.rs"))
        .expect("read arch_caps.rs");
    let code = source.split("#[cfg(test)]").next().unwrap();
    let mut atoms: Vec<&str> = code
        .lines()
        .filter_map(|line| line.trim().strip_prefix("let is_gfx"))
        .filter_map(|rest| rest.split_once(" = arch == \"gfx").map(|(_, tail)| tail))
        .filter_map(|tail| tail.strip_suffix("\";"))
        .collect();
    atoms.sort_unstable();
    let mut listed: Vec<&str> = ARCHES.iter().map(|arch| &arch["gfx".len()..]).collect();
    listed.sort_unstable();
    assert_eq!(
        listed, atoms,
        "ARCHES must list exactly the gfx atoms ArchCaps::new knows"
    );
}

#[test]
fn caps_accessor_list_is_every_public_accessor() {
    let source = std::fs::read_to_string(repo_root().join("crates/rdna-compute/src/arch_caps.rs"))
        .expect("read arch_caps.rs");
    let code = source.split("#[cfg(test)]").next().unwrap();
    let mut declared: Vec<&str> = code
        .lines()
        .filter_map(|line| {
            let rest = line.trim().strip_prefix("pub fn ")?;
            let (name, tail) = rest.split_once('(')?;
            tail.starts_with("&self)").then_some(name)
        })
        .filter(|name| !matches!(*name, "arch" | "should_use_mmq"))
        .collect();
    declared.sort_unstable();
    let mut listed: Vec<&str> = CAPS_ACCESSOR_NAMES.to_vec();
    listed.sort_unstable();
    assert_eq!(
        listed, declared,
        "caps_accessors! must list every ArchCaps accessor"
    );
}

#[test]
#[ignore = "regenerates tests/golden/flags-defaults/*.tsv"]
fn regenerate_flags_defaults_golden() {
    std::fs::create_dir_all(golden_dir()).unwrap();
    for (name, text) in render_files() {
        std::fs::write(golden_dir().join(name), text).unwrap();
    }
}

fn first_diff(golden: &str, actual: &str) -> String {
    let (mut g, mut a) = (golden.lines(), actual.lines());
    let mut line = 1;
    loop {
        match (g.next(), a.next()) {
            (Some(x), Some(y)) if x == y => line += 1,
            (None, None) => return "trailing bytes differ".to_owned(),
            (x, y) => {
                return format!(
                    "first difference at line {line}:\n  golden: {x:?}\n  actual: {y:?}"
                )
            }
        }
    }
}
