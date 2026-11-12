//! Classification of files in a local models directory.
//!
//! Shared by the CLI (`hipfire list`, `/v1/models`) and the TUI so both agree
//! on which on-disk files are models, which are sidecars of a model, and which
//! can serve a completion. Everything here is pure apart from the 32-byte
//! container-header read in [`probe_hfq_arch_id`].
//!
//! Functions that consult the grandfathered monolithics take the pin table as
//! a `&[GrandfatheredMonolithic]` so the caller chooses it (the CLI substitutes
//! a fixture in tests; the TUI passes [`crate::GRANDFATHERED_MONOLITHICS`]).

use crate::{
    grandfathered_monolithic_in, is_ple_format_token, GrandfatheredMonolithic, RegistryV1, Sidecar,
    EP_EXPERT_HOTSET_PINS, EXPERT_HOTSET_PINS, HC8_SIDECAR_PINS, HOT_UNITS_SIDECAR_PINS,
    RETIRED_MODELS, TRUNK_OVERLAY_PINS,
};
use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub const MODEL_SUFFIXES: &[&str] = &[
    ".hf4",
    ".hf6",
    ".hfq",
    ".mq2",
    ".mq2lloyd",
    // qt51 `MQ2G256LloydU` — the UNROTATED sibling of `.mq2lloyd` (qt19, which
    // is FWHT-rotated). Deliberately NOT folded into `.mq2lloyd`: loading a
    // rotated container as unrotated (or the reverse) does not fail loudly, it
    // produces silent garbage, so the two must stay nameable apart.
    ".mq2lloydu",
    ".mq2r",
    ".mq2rxt",
    ".mq3",
    ".mq3p",
    ".mq4",
    ".mq4p",
    ".mq4r",
    ".mq5",
    ".mq6",
    ".mfp4",
    ".q8",
];

/// Suffixes of PLE-adjacent files that belong to a model but are never models
/// themselves (besides the `.ple-<key>` row files, see [`is_ple_sidecar_name`]).
pub const PLE_AUX_SUFFIXES: [&str; 2] = [".plehot", ".plehotrows"];

/// HFQ arch ids a `hipfire serve` completion can be answered from — the
/// "Primary model ids" column of `docs/architecture-ids.md`, minus the
/// image-generation component trunks.
///
/// The polarity is deliberate. This predicate exists to keep `/v1/models` to
/// its contract ("every advertised id can answer a completion"), and it only
/// ever judges files the registry does not vouch for (registry entries short
/// out in [`is_standalone_model`]). An allowlist is the fail-closed direction:
/// an arch id that is not listed here — a sidecar the table gains later (a new
/// drafter id, a new FLUX component) or a container that is not a model at all
/// — is not advertised, whereas a denylist would silently re-open the reported
/// bug the day a new draft arch ships. The cost is that a brand-new *primary*
/// arch id is hidden from local discovery until this table is updated, which
/// the same commit has to do anyway: assigning an arch id means editing
/// `docs/architecture-ids.md` and the loader's carrier registry.
///
/// `40`/`45` (FLUX MMDiT trunks) are absent on purpose: they are image
/// generation components, served through `hipfire img` / `/v1/images/*`, not
/// completion trunks a `/v1/models` pick can generate text from. `8`
/// (dots.ocr) and `11` (LFM2.5 / LFM2.5-VL) are present: both answer text
/// completions. `16` (Qwen4 / Flash-Next) is present as a text-serving primary
/// arch: an unregistered local Flash-Next trunk must be advertised too.
pub const SERVE_TRUNK_ARCH_IDS: &[u32] = &[0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

/// Header-only read of an HFQ container's `arch_id`: the 32-byte container
/// header, no mmap, no metadata JSON parse, no tensor index.
///
/// A full open walks the metadata and materialises every tensor entry, which
/// is far too much work for a discovery scan. Serve's `/v1/models` uses this
/// to classify an on-disk file the registry does not vouch for — a trunk
/// versus a sidecar artifact (the DFlash drafts and friends carry arch ids
/// 20/21/22/23; see `docs/architecture-ids.md` § Sidecar / reserved ids).
/// `Err` for anything that is not a container, or is shorter than one, so a
/// caller can read "cannot tell" as "not a trunk".
///
/// The layout is the one the HFQ loader parses: `HFQM` magic, `u32` version,
/// `u32` arch id, `u32` tensor count, then the metadata and data offsets —
/// little-endian, relative to the container start.
pub fn probe_hfq_arch_id(path: &Path) -> std::io::Result<u32> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 32];
    file.read_exact(&mut header)?;
    if &header[..4] != b"HFQM" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("HfqFile: not an HFQ container: {}", path.display()),
        ));
    }
    Ok(u32::from_le_bytes(header[8..12].try_into().unwrap()))
}

/// True when `name` ends in one of [`MODEL_SUFFIXES`] (case-insensitive).
pub fn is_model_file(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    MODEL_SUFFIXES.iter().any(|suffix| lower.ends_with(suffix))
}

/// True for files that belong to a model without being one: the PLE row
/// sidecars of any format key (`.ple-q4`, `.ple-q8`, a future `.ple-q2`), the
/// hot-rank file and the hot-row packs.
pub fn is_ple_sidecar_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    PLE_AUX_SUFFIXES
        .iter()
        .any(|suffix| lower.ends_with(suffix))
        || lower
            .rsplit_once(".ple-")
            .is_some_and(|(stem, key)| !stem.is_empty() && is_ple_format_token(key))
}

/// True for a Flash-Next trunk overlay sidecar (`<base>.trunk-overlay`): an
/// HFQM container of pre-encoded QT44 tensors that belongs to its base and is
/// never a model on its own.
pub fn is_trunk_overlay_name(name: &str) -> bool {
    name.to_ascii_lowercase()
        .ends_with(hipfire_config::memory::TRUNK_OVERLAY_SIDECAR_SUFFIX)
}

/// True for a Flash-Next HC8 sidecar (`<base>.hc8`): an HFQM container of the
/// base's E4M3-requantized hyper-connection mixer weights that belongs to its
/// base and is never a model on its own.
pub fn is_hc8_sidecar_name(name: &str) -> bool {
    name.to_ascii_lowercase()
        .ends_with(hipfire_config::memory::HC8_SIDECAR_SUFFIX)
}

/// True for a Flash-Next EP2 hot-unit loan sidecar (`<base>.ep.hot-units`): calibration JSON
/// that belongs to its base and is never a model on its own.
pub fn is_hot_units_sidecar_name(name: &str) -> bool {
    name.to_ascii_lowercase()
        .ends_with(hipfire_config::memory::EP_HOT_UNITS_SIDECAR_SUFFIX)
}

/// Expert placement metadata belongs to its base, never to model discovery.
pub fn is_expert_hotset_name(name: &str) -> bool {
    name.to_ascii_lowercase()
        .ends_with(hipfire_config::memory::EXPERT_HOTSET_SIDECAR_SUFFIX)
}

/// Every file the registry names as a *sidecar* of some entry — `dflash`,
/// `vision`, `mtp`, `dspark`, the FLUX components, and the
/// alternative `heads` carriers.
///
/// These share the model suffixes (`.hfq`), so a suffix test alone cannot
/// tell them apart from a trunk. A sidecar is not a loadable model on its
/// own: `qwen3.8-27b-vision.hfq` carries only tower tensors and embeds
/// `"tokenizer": "{}"`, so loading it as a trunk fails with
/// `tokenizer metadata field missing or wrong type: model`.
pub fn registry_sidecar_files(registry: &RegistryV1) -> BTreeSet<&str> {
    let mut files: BTreeSet<&str> = registry
        .models
        .values()
        .flat_map(|entry| {
            [
                entry.mtp.as_ref(),
                entry.dspark.as_ref(),
                entry.t5.as_ref(),
                entry.clip.as_ref(),
                entry.qwen3.as_ref(),
                entry.vae.as_ref(),
                entry.dflash.as_ref(),
                entry.vision.as_ref(),
            ]
            .into_iter()
            .flatten()
            .map(|sidecar: &Sidecar| sidecar.file.as_str())
            .chain(entry.heads.values().map(|sidecar| sidecar.file.as_str()))
            .chain(entry.ple.values().map(|sidecar| sidecar.file.as_str()))
        })
        .collect();
    // These sidecars are compiled-in pins, not registry fields.
    files.extend(TRUNK_OVERLAY_PINS.iter().map(|pin| pin.file));
    files.extend(HC8_SIDECAR_PINS.iter().map(|pin| pin.file));
    files.extend(HOT_UNITS_SIDECAR_PINS.iter().map(|pin| pin.file));
    files.extend(EXPERT_HOTSET_PINS.iter().map(|pin| pin.file));
    files.extend(EP_EXPERT_HOTSET_PINS.iter().map(|pin| pin.file));
    files
}

/// True when `name` is some registry entry's own `file`, a spelling of the
/// pinned monolithic a split entry is grandfathered with in `pins` (a verified
/// local install keeps serving, so its files are the registry's models), or the
/// file of a compiled [`RETIRED_MODELS`] pin (a retired model still runs from a
/// verified local copy, though the catalog no longer lists it).
pub fn names_registry_model(
    name: &str,
    registry: &RegistryV1,
    pins: &[GrandfatheredMonolithic],
) -> bool {
    RETIRED_MODELS.iter().any(|pin| pin.file == name)
        || registry.models.values().any(|entry| {
            entry.file == name
                || grandfathered_monolithic_in(entry, pins)
                    .is_some_and(|pin| name == pin.canonical_file || name == pin.legacy_file)
        })
}

/// True when `name` is a model file worth showing in a local listing.
///
/// A model suffix OR being some registry entry's own `file`. The second term
/// is load-bearing: several registry tiers ship files whose suffix is absent
/// from [`MODEL_SUFFIXES`] (`qwen3.8-27b.mq3-xt`, `-pro`, `.bq1`, `.tq2`), and a
/// suffix-only test hides exactly those models.
pub fn is_listable_model_file(
    name: &str,
    registry: &RegistryV1,
    pins: &[GrandfatheredMonolithic],
) -> bool {
    !is_ple_sidecar_name(name)
        && !is_trunk_overlay_name(name)
        && !is_hc8_sidecar_name(name)
        && !is_hot_units_sidecar_name(name)
        && !is_expert_hotset_name(name)
        && (is_model_file(name) || names_registry_model(name, registry, pins))
}

/// True when `name` can actually serve a completion, as opposed to merely
/// being a model-shaped file on disk.
///
/// This is the `/v1/models` discovery predicate and it is stricter than
/// [`is_listable_model_file`] on purpose. A file the registry names in ANY
/// sidecar slot is not a serve target, even when some entry also lists it as
/// its own `file` — that overlap is exactly the DFlash drafts
/// (`qwen38-27b-dflash-mq3.hfq` and friends), pullable by tag but arch
/// 20/22/23 artifacts that `docs/architecture-ids.md` classes as "not primary
/// `load_model` trunk targets". The vision tower (`qwen3.8-27b-vision.hfq`) is
/// sidecar-only and embeds `"tokenizer": "{}"`, so selecting it loads all 64
/// layers and then fails with `tokenizer metadata field missing or wrong
/// type: model`.
///
/// Advertising either in `/v1/models` hands an OpenAI-compatible client an id
/// that cannot answer. Both stay pullable and resolvable by tag, and both
/// still appear in `hipfire list` — only the serve surface drops them.
///
/// A file the registry does not name at all (a locally quantized trunk, or a
/// locally produced draft such as `qwen3-8b-dflash.hfq`) is classified by its
/// own container header: `/v1/models` must not advertise it unless
/// [`probe_hfq_arch_id`] reports a [`SERVE_TRUNK_ARCH_IDS`] id. The registry
/// cannot answer for it — the catalog only knows the files it ships.
pub fn is_standalone_model(
    path: &Path,
    name: &str,
    registry: &RegistryV1,
    pins: &[GrandfatheredMonolithic],
) -> bool {
    if is_ple_sidecar_name(name) || registry_sidecar_files(registry).contains(name) {
        return false;
    }
    if !is_listable_model_file(name, registry, pins) {
        return false;
    }
    if names_registry_model(name, registry, pins) {
        return true;
    }
    probe_hfq_arch_id(path).is_ok_and(|arch_id| SERVE_TRUNK_ARCH_IDS.contains(&arch_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GRANDFATHERED_MONOLITHICS;
    use std::path::PathBuf;

    /// Scratch directory removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "hipfire-registry-local-files-{label}-{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn join(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A 32-byte container header carrying `arch_id`.
    fn write_header(path: &Path, arch_id: u32) {
        let mut header = [0u8; 32];
        header[..4].copy_from_slice(b"HFQM");
        header[4..8].copy_from_slice(&1u32.to_le_bytes());
        header[8..12].copy_from_slice(&arch_id.to_le_bytes());
        std::fs::write(path, header).unwrap();
    }

    #[test]
    fn model_suffix_filter_covers_current_formats() {
        assert!(is_model_file("qwen3.6-35b-a3b.mq4r"));
        assert!(is_model_file("deepseek.mq2lloyd"));
        assert!(is_model_file("deepseek-v4-flash-0731.mq2r"));
        assert!(is_model_file("deepseek-v4-flash-0731.mq2rxt"));
        assert!(is_model_file("draft.hfq"));
        assert!(!is_model_file("README.md"));
    }

    #[test]
    fn sidecar_classification_covers_rows_hot_rank_and_hot_packs() {
        for name in [
            "m.mq4-xts.ple-q4",
            "m.mq4-xts.ple-q8",
            "M.MQ4-XTS.PLE-Q8",
            "m.mq4-xts.ple-q2",
            "m.mq4-xts.ple-nf4",
            "m.mq4-xts.plehot",
            "m.mq4-xts.plehotrows",
            "m.mq4-xts.ple-q4.plehotrows",
            "m.mq4-xts.ple-q8.plehotrows",
        ] {
            assert!(is_ple_sidecar_name(name), "{name}");
        }
        for name in [
            "m.mq4-xts",
            "qwen3.8-flash-next-base.mq4-xts",
            "qwen3.8-flash-next.mq4",
            "m.q8",
            "ple-q8.mq4",
            "m.ple-",
            "m.ple-q_2",
            ".ple-q8",
        ] {
            assert!(!is_ple_sidecar_name(name), "{name}");
        }
    }

    #[test]
    fn hc8_sidecar_is_classified_listed_as_a_sidecar_and_never_a_model() {
        for name in [
            "qwen3.8-flash-next-base.mq4-xts-symtrunk.hc8",
            "M.MQ4-XTS.HC8",
        ] {
            assert!(is_hc8_sidecar_name(name), "{name}");
        }
        for name in [
            "qwen3.8-flash-next-base.mq4-xts-symtrunk",
            "m.hc8.mq4",
            "m.trunk-overlay",
        ] {
            assert!(!is_hc8_sidecar_name(name), "{name}");
        }
        let registry: RegistryV1 = serde_json::from_str(
            r#"{"schema_version":1,"generated_at":"t","models":{},"aliases":{}}"#,
        )
        .unwrap();
        let pin = &HC8_SIDECAR_PINS[0];
        assert!(!is_listable_model_file(
            pin.file,
            &registry,
            GRANDFATHERED_MONOLITHICS
        ));
        assert!(registry_sidecar_files(&registry).contains(pin.file));
    }

    #[test]
    fn hot_units_sidecar_is_classified_listed_as_a_sidecar_and_never_a_model() {
        for name in [
            "qwen3.8-flash-next-base.mq4-xts-symtrunk.ep.hot-units",
            "M.MQ4-XTS.EP.HOT-UNITS",
        ] {
            assert!(is_hot_units_sidecar_name(name), "{name}");
        }
        for name in [
            "qwen3.8-flash-next-base.mq4-xts-symtrunk",
            "m.ep.expert-hotset",
            "m.hot-units.mq4",
        ] {
            assert!(!is_hot_units_sidecar_name(name), "{name}");
        }
        let registry: RegistryV1 = serde_json::from_str(
            r#"{"schema_version":1,"generated_at":"t","models":{},"aliases":{}}"#,
        )
        .unwrap();
        let pin = &HOT_UNITS_SIDECAR_PINS[0];
        assert!(!is_listable_model_file(
            pin.file,
            &registry,
            GRANDFATHERED_MONOLITHICS
        ));
        assert!(registry_sidecar_files(&registry).contains(pin.file));
    }

    /// `/v1/models` is the discovery surface an OpenAI-compatible client reads
    /// to choose a model. It must offer only files that load as a trunk.
    ///
    /// Two ways the old suffix-only scan got that wrong, both observed against
    /// a real `~/.hipfire/models`:
    ///   - it ADVERTISED `qwen3.8-27b-vision.hfq` and `qwen38-27b-dflash-mq3.hfq`,
    ///     which are sidecars; selecting either fails at load with
    ///     `tokenizer metadata field missing or wrong type: model`.
    ///   - it DROPPED `qwen3.8-27b.mq3-xt`, whose `-xt` suffix is not in
    ///     `MODEL_SUFFIXES`, so the tier the user actually wanted was invisible.
    #[test]
    fn standalone_model_classification_separates_sidecars_from_tiers() {
        let registry = crate::bundled().unwrap();
        let pins = GRANDFATHERED_MONOLITHICS;
        // Registry-vouched files are judged by the catalog, never by the file:
        // these paths deliberately do not exist.
        let classified = |name: &str| {
            is_standalone_model(
                &PathBuf::from("/nonexistent/models").join(name),
                name,
                &registry,
                pins,
            )
        };

        // The tier whose suffix is missing from MODEL_SUFFIXES: listable AND
        // servable. This is the model the operator actually wants.
        assert!(!is_model_file("qwen3.8-27b.mq3-xt"));
        assert!(is_listable_model_file(
            "qwen3.8-27b.mq3-xt",
            &registry,
            pins
        ));
        assert!(classified("qwen3.8-27b.mq3-xt"));

        // A sidecar-only file: listable (it is really on disk) but NOT
        // servable. It embeds `"tokenizer": "{}"`.
        assert!(is_listable_model_file(
            "qwen3.8-27b-vision.hfq",
            &registry,
            pins
        ));
        assert!(!classified("qwen3.8-27b-vision.hfq"));

        // The overlap case: the DFlash draft is BOTH its own registry entry's
        // `file` (the documented `hipfire pull qwen3.8:27b-draft-mq3` target)
        // AND a `dflash` sidecar of three other entries. Arch 20 is not a serve
        // trunk, so it must not be advertised — while staying pullable.
        assert!(registry
            .models
            .values()
            .any(|entry| entry.file == "qwen38-27b-dflash-mq3.hfq"));
        assert!(is_listable_model_file(
            "qwen38-27b-dflash-mq3.hfq",
            &registry,
            pins
        ));
        assert!(!classified("qwen38-27b-dflash-mq3.hfq"));

        assert!(!is_listable_model_file("README.md", &registry, pins));
        assert!(!classified("README.md"));
    }

    /// A file the registry does not name at all — what `hipfire quantize`
    /// leaves in the models dir — is classified by its own container header.
    ///
    /// This is the class the registry-sidecar filter cannot see: on a real
    /// box, `/v1/models` advertised `qwen3-8b-dflash.hfq`, and selecting it
    /// failed with `no carrier for HFQ arch_id=20`.
    #[test]
    fn unregistered_files_are_classified_by_their_container_arch_id() {
        let registry = crate::bundled().unwrap();
        let dir = TempDir::new("unregistered-arch");
        let classify = |file: &str| {
            let path = dir.join(file);
            is_standalone_model(&path, file, &registry, GRANDFATHERED_MONOLITHICS)
        };

        // A locally quantized trunk of a primary arch: advertised.
        write_header(&dir.join("local-trunk.mq4"), 5);
        assert!(classify("local-trunk.mq4"));

        // A locally produced Flash-Next (Qwen4) trunk, arch 16, is a text-serving
        // primary: advertised even though the registry does not name the file.
        write_header(&dir.join("local-flash-next.mq4"), 16);
        assert!(classify("local-flash-next.mq4"));

        // Locally produced sidecars: never advertised, whatever the registry
        // knows. 20 = DFlash draft, 23 = Muse Glimmer drafter.
        write_header(&dir.join("qwen3-8b-dflash.hfq"), 20);
        write_header(&dir.join("glimmer-assistant.hfq"), 23);
        assert!(!classify("qwen3-8b-dflash.hfq"));
        assert!(!classify("glimmer-assistant.hfq"));

        // Not a container at all (or truncated): fail closed rather than
        // advertise an id that cannot be classified.
        std::fs::write(dir.join("garbage.hfq"), b"not an HFQ container").unwrap();
        std::fs::write(dir.join("short.hfq"), b"HFQM\x01\x00\x00\x00").unwrap();
        assert!(!classify("garbage.hfq"));
        assert!(!classify("short.hfq"));

        // A FLUX component trunk is a container with a carrier, but it cannot
        // answer a completion: `/v1/models` is the completion surface.
        write_header(&dir.join("flux-transformer.hfq"), 40);
        assert!(!classify("flux-transformer.hfq"));
    }

    #[test]
    fn probe_reads_the_arch_id_of_a_container() {
        let dir = TempDir::new("probe-ok");
        write_header(&dir.join("trunk.hfq"), 5);
        assert_eq!(probe_hfq_arch_id(&dir.join("trunk.hfq")).unwrap(), 5);
        // The sidecar class the discovery filter exists for: a DFlash draft.
        write_header(&dir.join("draft.hfq"), 20);
        assert_eq!(probe_hfq_arch_id(&dir.join("draft.hfq")).unwrap(), 20);
    }

    #[test]
    fn probe_fails_closed_on_a_non_container() {
        let dir = TempDir::new("probe-bad");
        std::fs::write(
            dir.join("README.md"),
            b"# not a model\n# padding to 32 bytes....\n",
        )
        .unwrap();
        assert!(probe_hfq_arch_id(&dir.join("README.md")).is_err());
        // Shorter than the 32-byte header: `Err`, not a panic.
        std::fs::write(dir.join("truncated.hfq"), b"HFQM\x01\x00\x00\x00").unwrap();
        assert!(probe_hfq_arch_id(&dir.join("truncated.hfq")).is_err());
        assert!(probe_hfq_arch_id(&dir.join("absent.hfq")).is_err());
    }
}
