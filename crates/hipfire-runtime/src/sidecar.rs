// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sidecar discovery for a trunk model file.
//!
//! hipfire stores optional per-model companions next to the trunk:
//!
//! * `<stem>.mtp` — the MTP draft head (`hipfire-arch-qwen35::mtp_head`)
//! * `<stem>.vl`  — the vision tower (`.vl` carrier)
//!
//! `stem` is not simply the trunk's last extension: production trunks are
//! quant-tagged (`qwen3.8-27b.mq4v2.hfq`), so the companion is named after
//! the model, not after its quant suffix (`qwen3.8-27b.mtp`). Every hop that
//! resolves a companion MUST use this one candidate list — three independent
//! copies of the probe already existed (daemon, loader, CLI capability
//! advertisement) and the `.vl` copies had dropped the quant-suffix stripping
//! the `.mtp` copy had, so a quant-tagged trunk silently lost its vision
//! tower while keeping its draft head.

use std::path::{Path, PathBuf};

/// Quant/build suffixes hipfire writes into model filenames. Stripping one
/// maps `model.<quant>.hfq` to the companion's `model.<ext>`.
const QUANT_SUFFIXES: &[&str] = &[
    ".mq4v2",
    ".mq6v2",
    ".mq5v2",
    ".mq3v2",
    ".mq2v2",
    ".mq4cg256",
    ".mq4",
    ".mq6",
    ".mq8",
    ".q8",
    ".q4",
    ".bf16",
];

/// `Path` extension of a companion file (`"mtp"`, `"vl"`).
pub type SidecarExt = &'static str;

/// Ordered, de-duplicated `<base>.<ext>` candidates for `trunk`, most
/// specific first:
///
/// 1. the trunk with its last extension replaced (`model.mq4v2.hfq` →
///    `model.mq4v2.mtp`);
/// 2. the trunk with `.hfq` and each known quant suffix stripped
///    (`model.mtp`);
/// 3. the trunk with any trailing alphanumeric-only extension stripped
///    (a generic last resort for suffixes added after this list was
///    written).
///
/// The first candidate that exists on disk wins, so a more specific name
/// always shadows a less specific one.
pub fn sidecar_candidates(trunk: &Path, ext: SidecarExt) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if !out.iter().any(|e| e == &p) {
            out.push(p);
        }
    };
    push(trunk.with_extension(ext));
    let name = trunk.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let parent = trunk.parent().unwrap_or_else(|| Path::new("."));
    let stem = name.strip_suffix(".hfq").unwrap_or(name);
    for q in QUANT_SUFFIXES {
        if let Some(base) = stem.strip_suffix(*q) {
            push(parent.join(format!("{base}.{ext}")));
        }
    }
    if let Some((base, rest)) = stem.rsplit_once('.') {
        if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_alphanumeric()) {
            push(parent.join(format!("{base}.{ext}")));
        }
    }
    out
}

/// `.vl` (vision tower) candidates for `trunk`, most specific first.
pub fn vl_sidecar_candidates(trunk: &Path) -> Vec<PathBuf> {
    sidecar_candidates(trunk, "vl")
}

/// First existing `.vl` sibling of `trunk`, ignoring the `HIPFIRE_VL_FILE`
/// override. Used where only the on-disk sibling convention matters.
pub fn find_vl_sidecar(trunk: &Path) -> Option<PathBuf> {
    vl_sidecar_candidates(trunk)
        .into_iter()
        .find(|p| p.exists())
}

/// Resolve the vision tower for `trunk` exactly as a model load does:
/// `HIPFIRE_VL_FILE` first (an explicit override that does not exist is a
/// typo and is reported), then the `<stem>.vl` sibling candidates.
pub fn resolve_vl_sidecar(model_path: &str) -> Option<PathBuf> {
    if let Ok(v) = hipfire_config::developer_var("HIPFIRE_VL_FILE") {
        let p = PathBuf::from(&v);
        if p.exists() {
            return Some(p);
        }
        if !v.trim().is_empty() {
            eprintln!(
                "HIPFIRE_VL_FILE is set to {v:?} but the file does not exist; \
                 falling back to <stem>.vl sibling discovery"
            );
        }
    }
    find_vl_sidecar(Path::new(model_path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// One temp dir per test process; the tests never share it.
    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("hipfire-sidecar-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    #[test]
    fn a_quant_suffixed_trunk_finds_its_stripped_siblings() {
        let dir = scratch("quant");
        let trunk = dir.join("model.mq4v2.hfq");
        fs::write(&trunk, b"x").unwrap();
        // Production layout: the companion is named after the model.
        let vl = dir.join("model.vl");
        let mtp = dir.join("model.mtp");
        fs::write(&vl, b"x").unwrap();
        fs::write(&mtp, b"x").unwrap();
        assert_eq!(find_vl_sidecar(&trunk), Some(vl.clone()));
        assert_eq!(
            sidecar_candidates(&trunk, "mtp")
                .into_iter()
                .find(|p| p.exists()),
            Some(mtp)
        );
        // A more specific name still wins over the stripped one.
        let exact = dir.join("model.mq4v2.vl");
        fs::write(&exact, b"x").unwrap();
        assert_eq!(find_vl_sidecar(&trunk), Some(exact));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unknown_tagged_trunk_still_finds_the_generic_sibling() {
        let dir = scratch("generic");
        let trunk = dir.join("model.newquant.hfq");
        fs::write(&trunk, b"x").unwrap();
        let vl = dir.join("model.vl");
        fs::write(&vl, b"x").unwrap();
        assert_eq!(find_vl_sidecar(&trunk), Some(vl));
        let _ = fs::remove_dir_all(&dir);
    }
}
