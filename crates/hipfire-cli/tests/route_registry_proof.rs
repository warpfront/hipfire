// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! 0.5.0 route-selection proof (GPU-free): the retained-Redline route chosen
//! from file identity (registry card `quant`, then HFQ metadata) equals the
//! route the removed file-extension rule chose, for every artifact of the
//! compiled registry, retired models included, on every admission input.
//!
//! `local_files_route_proof` (ignored; run it explicitly) additionally reads
//! the HFQ headers of the registry artifacts present on this machine and
//! writes the table to `$ROUTE_PROOF_OUT`.

use hipfire_registry::{ModelEntry, RETIRED_MODELS};
use hipfire_runtime::config::retained_redline_default;
use hipfire_runtime::route_sku::{self, RouteSku};
use std::path::{Path, PathBuf};

/// Every registry artifact: the bundled catalog, then the retired pins.
fn all_artifacts() -> Vec<(String, ModelEntry)> {
    let registry = hipfire_registry::bundled().expect("bundled registry");
    let mut artifacts: Vec<(String, ModelEntry)> = registry
        .models
        .iter()
        .map(|(tag, entry)| (tag.clone(), entry.clone()))
        .collect();
    artifacts.extend(
        RETIRED_MODELS
            .iter()
            .map(|pin| (format!("{} (retired)", pin.tag), pin.entry())),
    );
    artifacts
}

/// The route the removed extension rule gave a file name.
fn legacy_sku(file: &str) -> Option<RouteSku> {
    let extension = Path::new(file)
        .extension()
        .and_then(|extension| extension.to_str())?;
    [
        ("mq4r", RouteSku::Mq4r),
        ("mq4rs", RouteSku::Mq4rs),
        ("mq2r", RouteSku::Mq2r),
    ]
    .into_iter()
    .find_map(|(name, sku)| extension.eq_ignore_ascii_case(name).then_some(sku))
}

/// The removed `retained_redline_default`, extension rules and all, verbatim
/// except that the process-config CPU-offload check (path independent, and
/// unset in a test process) is omitted from both sides.
fn legacy_retained_redline_default(
    gpu_arch: &str,
    model_arch: &str,
    model_path: &str,
    pp: usize,
    tp: usize,
    ep: usize,
    has_drafter: bool,
) -> bool {
    let ext = |wanted: &str| {
        Path::new(model_path)
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case(wanted))
    };
    if ep != 1 {
        return false;
    }
    if model_arch.eq_ignore_ascii_case("muse_glimmer") || model_arch.eq_ignore_ascii_case("gemma4")
    {
        return false;
    }
    if model_arch.eq_ignore_ascii_case("qwen4") && matches!(gpu_arch, "gfx1100" | "gfx1151") {
        return false;
    }
    if gpu_arch == "gfx1100"
        && model_arch.eq_ignore_ascii_case("qwen3_5")
        && pp == 1
        && tp == 1
        && !has_drafter
        && ext("mq3-xt")
    {
        return true;
    }
    if matches!(gpu_arch, "gfx1100" | "gfx1151" | "gfx1201") && pp == 1 && tp == 1 && ext("mq4r") {
        return true;
    }
    if gpu_arch.eq_ignore_ascii_case("gfx1201")
        && (model_arch.eq_ignore_ascii_case("qwen3_5") || model_arch.eq_ignore_ascii_case("qwen4"))
        && pp == 1
        && tp == 1
        && !has_drafter
    {
        return true;
    }
    if gpu_arch.eq_ignore_ascii_case("gfx1201")
        && model_arch.eq_ignore_ascii_case("qwen3_5_moe")
        && pp == 1
        && tp == 1
        && !has_drafter
        && ext("mq4rs")
    {
        return true;
    }
    gpu_arch.eq_ignore_ascii_case("gfx1151")
        && model_arch.eq_ignore_ascii_case("deepseek4")
        && pp == 1
        && tp == 1
        && !has_drafter
        && ext("mq2r")
}

fn all_model_arches() -> Vec<&'static str> {
    let mut labels: Vec<&'static str> = (0..64).map(hipfire_loader::arch_label).collect();
    labels.sort_unstable();
    labels.dedup();
    labels
}

#[test]
fn registry_identity_route_equals_the_removed_extension_route_for_every_artifact() {
    let arches = all_model_arches();
    let artifacts = all_artifacts();
    assert!(artifacts.len() > 90, "{} artifacts", artifacts.len());
    let mut sku_rows = 0;
    for (tag, entry) in &artifacts {
        let old = legacy_sku(&entry.file);
        let new = route_sku::registry_route(entry.quant.as_deref());
        assert_eq!(old, new, "{tag} ({}): old vs new SKU", entry.file);
        sku_rows += usize::from(new.is_some());
        for gpu_arch in [
            "gfx1030", "gfx1100", "gfx1101", "gfx1151", "gfx1200", "gfx1201", "gfx942",
        ] {
            for model_arch in &arches {
                for (pp, tp, ep) in [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2)] {
                    for has_drafter in [false, true] {
                        let was = legacy_retained_redline_default(
                            gpu_arch,
                            model_arch,
                            &entry.file,
                            pp,
                            tp,
                            ep,
                            has_drafter,
                        );
                        let is = retained_redline_default(
                            gpu_arch,
                            model_arch,
                            &entry.file,
                            new,
                            pp,
                            tp,
                            ep,
                            has_drafter,
                        );
                        assert_eq!(
                            was, is,
                            "{tag} ({}) {gpu_arch} {model_arch} pp={pp} tp={tp} ep={ep} \
                             drafter={has_drafter}",
                            entry.file
                        );
                    }
                }
            }
        }
    }
    // The registry really carries the SKUs this proves (MQ4R x3, MQ4RS, MQ2R).
    assert_eq!(sku_rows, 5, "registry SKU artifacts");
}

/// The cards behind each SKU, so the ROUTE-PROOF table and this test cannot
/// drift from the registry.
#[test]
fn registry_sku_artifacts_are_exactly_the_five_known_cards() {
    let mut cards: Vec<(String, RouteSku)> = all_artifacts()
        .into_iter()
        .filter_map(|(tag, entry)| {
            route_sku::registry_route(entry.quant.as_deref()).map(|sku| (tag, sku))
        })
        .collect();
    cards.sort_by(|a, b| a.0.cmp(&b.0));
    let tags: Vec<&str> = cards.iter().map(|(tag, _)| tag.as_str()).collect();
    assert_eq!(
        tags,
        [
            "deepseek-v4-flash",
            "muse-glimmer:fast",
            "ornith-1.5:35b-a3b-mq4r",
            "qwen3.6:35b-a3b-mq4r",
            "qwen3.6:35b-a3b-mq4rs",
        ]
    );
}

fn local_model_dirs() -> Vec<PathBuf> {
    let home = PathBuf::from(std::env::var("HOME").expect("HOME"));
    vec![
        home.join(".hipfire/models"),
        home.join(".hipfire-zed-live-gfx1201/models"),
        home.join("hipfire-pr-queue/home/models"),
        home.join("qcal/models"),
    ]
}

/// Reads the header of every registry artifact found on this machine, runs the
/// selector on it, and writes the proof table to `$ROUTE_PROOF_OUT`.
#[test]
#[ignore = "reads local model headers; writes $ROUTE_PROOF_OUT"]
fn local_files_route_proof() {
    let out = std::env::var("ROUTE_PROOF_OUT").expect("ROUTE_PROOF_OUT");
    let mut table = String::from(
        "| registry tag | file | quant | old route (extension) | new route (registry identity) | local file check |\n\
         |---|---|---|---|---|---|\n",
    );
    let dirs = local_model_dirs();
    let mut checked = 0;
    for (tag, entry) in all_artifacts() {
        let old = legacy_sku(&entry.file);
        let new = route_sku::registry_route(entry.quant.as_deref());
        assert_eq!(old, new, "{tag}");
        let local = dirs.iter().map(|dir| dir.join(&entry.file)).find(|path| {
            std::fs::metadata(path)
                .map(|meta| Some(meta.len()) == entry.size_bytes)
                .unwrap_or(false)
        });
        let check = match local {
            None => "not present locally (route proved from the registry card only)".to_owned(),
            Some(path) => {
                checked += 1;
                let hfq = hipfire_runtime::hfq::HfqFile::open_at_offset(&path, 0)
                    .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
                let header = route_sku::header_route(&hfq.metadata_json)
                    .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
                let resolved = route_sku::resolve(&hfq, entry.quant.as_deref())
                    .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
                assert_eq!(
                    resolved.map(|route| route.sku),
                    old,
                    "{tag}: resolved route of {}",
                    path.display()
                );
                format!(
                    "`{}` size matches; HFQ route key: {}; resolved {} via {}",
                    path.display(),
                    header.map_or("absent".to_owned(), |sku| sku.as_str().to_owned()),
                    resolved.map_or("none", |route| route.sku.as_str()),
                    resolved.map_or("-", |route| route.source.as_str()),
                )
            }
        };
        table.push_str(&format!(
            "| {tag} | {} | {} | {} | {} | {check} |\n",
            entry.file,
            entry.quant.as_deref().unwrap_or("-"),
            old.map_or("none", RouteSku::as_str),
            new.map_or("none", RouteSku::as_str),
        ));
    }
    table.push_str(&format!(
        "\nartifacts checked against a local file: {checked}\n"
    ));
    std::fs::write(&out, table).unwrap();
}
