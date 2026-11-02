//! In-memory code-object identity shared by hipcc HSACO and embedded
//! PeaceMaker images.
//!
//! A successful module load admits one immutable [`CodeObjectArtifact`]: the
//! exact image bytes, their SHA-256 ([`CodeObjectId`]), the origin and the
//! optional Radiowave certification. Whole-image hashing, file reads and
//! manifest parsing happen once at admission; recording and replay only clone
//! the [`RecordedArtifact`] handle and never touch the filesystem. Lab-only
//! device-ELF identities are derived from those admitted bytes and cached once
//! per architecture on first observation.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use radiowave::CodeObjectCertification;
use redline_dispatch::aql::CodeObjectBytes;
use sha2::{Digest, Sha256};

/// SHA-256 of the exact registered image (whole ELF or HIP bundle).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CodeObjectId(pub [u8; 32]);

impl CodeObjectId {
    pub fn of(image: &[u8]) -> Self {
        Self(Sha256::digest(image).into())
    }

    pub fn to_hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

impl fmt::Debug for CodeObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CodeObjectId({})", self.to_hex())
    }
}

impl fmt::Display for CodeObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodeObjectOrigin {
    /// hipcc cache file the image was read from (diagnostic provenance only).
    HipFile(PathBuf),
    /// Embedded native image, named by its module.
    NativeEmbedded(&'static str),
}

/// Immutable admitted code object.
#[derive(Debug)]
pub struct CodeObjectArtifact {
    id: CodeObjectId,
    image: CodeObjectBytes,
    origin: CodeObjectOrigin,
    radiowave: Option<CodeObjectCertification>,
    /// Cached per-arch raw device IDs: SHA-256 of the `arch` device ELF
    /// extracted from the admitted image via `device_code_object`
    /// (raw ELF as is; the arch entry of an uncompressed clang offload
    /// bundle). `#[cfg(lab)]`-only; empty (no alloc) until lab uses it.
    /// `None` per arch means extraction refused (compressed/missing/
    /// ambiguous/malformed) — never the container digest as fallback.
    #[cfg(feature = "lab")]
    device_ids: std::sync::Mutex<std::collections::HashMap<String, Option<CodeObjectId>>>,
}

pub type RecordedArtifact = Arc<CodeObjectArtifact>;

impl CodeObjectArtifact {
    /// Admit a hipcc image already read from `path`. The adjacent
    /// `.radiowave.json` sidecar is verified once against these exact bytes;
    /// a missing, malformed or hash-stale sidecar leaves no certification.
    pub fn hip_file(path: &Path, image: Arc<[u8]>) -> Self {
        let radiowave = std::fs::read_to_string(path.with_extension("radiowave.json"))
            .ok()
            .and_then(|manifest| CodeObjectCertification::from_json(&image, &manifest).ok());
        Self {
            id: CodeObjectId::of(&image),
            image: image.into(),
            origin: CodeObjectOrigin::HipFile(path.to_path_buf()),
            radiowave,
            #[cfg(feature = "lab")]
            device_ids: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Admit an embedded native image. `radiowave_json` is verified against
    /// the image exactly as a file sidecar would be.
    pub fn native_embedded(
        module: &'static str,
        image: CodeObjectBytes,
        radiowave_json: Option<&str>,
    ) -> Self {
        let radiowave = radiowave_json.and_then(|manifest| {
            CodeObjectCertification::from_json(image.as_bytes(), manifest).ok()
        });
        Self {
            id: CodeObjectId::of(image.as_bytes()),
            image,
            origin: CodeObjectOrigin::NativeEmbedded(module),
            radiowave,
            #[cfg(feature = "lab")]
            device_ids: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    pub fn id(&self) -> CodeObjectId {
        self.id
    }

    /// Raw device ID for `arch`: SHA-256 of the `arch` device ELF extracted
    /// from the admitted image (`#[cfg(lab)]`-only; cached per arch, hashed
    /// once from the borrowed payload). `None` when extraction refuses
    /// (compressed/missing/ambiguous/malformed) — never the container digest.
    #[cfg(feature = "lab")]
    pub fn device_id(&self, arch: &str) -> Option<CodeObjectId> {
        let mut cached = self.device_ids.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(known) = cached.get(arch) {
            return *known;
        }
        let id =
            crate::scratch::device_code_object(self.image.as_bytes(), arch).map(CodeObjectId::of);
        cached.insert(arch.to_string(), id);
        id
    }

    pub fn image(&self) -> &CodeObjectBytes {
        &self.image
    }

    pub fn origin(&self) -> &CodeObjectOrigin {
        &self.origin
    }

    pub fn radiowave(&self) -> Option<&CodeObjectCertification> {
        self.radiowave.as_ref()
    }

    /// Recording provenance: legacy HIP keeps its cache path; native images
    /// report module and digest.
    pub fn provenance_json(&self) -> serde_json::Value {
        match &self.origin {
            CodeObjectOrigin::HipFile(path) => serde_json::json!({
                "origin": "hip_file",
                "path": path.display().to_string(),
                "image_sha256": self.id.to_hex(),
            }),
            CodeObjectOrigin::NativeEmbedded(module) => serde_json::json!({
                "origin": "native_embedded",
                "module": module,
                "image_sha256": self.id.to_hex(),
            }),
        }
    }

    /// Whether `image` is byte-for-byte this artifact's image; pointer
    /// equality short-circuits the hash for repeated static loads.
    fn holds(&self, image: &[u8]) -> bool {
        let own = self.image.as_bytes();
        (own.as_ptr() == image.as_ptr() && own.len() == image.len())
            || CodeObjectId::of(image) == self.id
    }
}

/// Artifacts are equal exactly when their images are (same digest).
impl PartialEq for CodeObjectArtifact {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for CodeObjectArtifact {}

/// Diagnostic label: the HIP cache path, or the native module and digest.
impl fmt::Display for CodeObjectArtifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.origin {
            CodeObjectOrigin::HipFile(path) => write!(f, "{}", path.display()),
            CodeObjectOrigin::NativeEmbedded(module) => {
                write!(f, "native:{module}@{}", &self.id.to_hex()[..16])
            }
        }
    }
}

/// Module and launched-function bindings to admitted artifacts. A name binds
/// to exactly one digest for the lifetime of the loaded module cache.
#[derive(Default)]
pub struct CodeObjectRegistry {
    modules: HashMap<String, RecordedArtifact>,
    functions: HashMap<String, RecordedArtifact>,
}

impl CodeObjectRegistry {
    /// Bind `module` to `artifact`; re-admitting the identical digest returns
    /// the existing handle, a different digest is rejected.
    pub fn admit_module(
        &mut self,
        module: &str,
        artifact: CodeObjectArtifact,
    ) -> Result<RecordedArtifact, String> {
        if let Some(bound) = self.modules.get(module) {
            return if bound.id == artifact.id {
                Ok(bound.clone())
            } else {
                Err(format!(
                    "module {module:?} is bound to image {} and cannot rebind to {}",
                    bound.id, artifact.id
                ))
            };
        }
        let artifact = Arc::new(artifact);
        self.modules.insert(module.to_owned(), artifact.clone());
        Ok(artifact)
    }

    /// Bind a launched function (one export of `artifact`).
    pub fn bind_function(&mut self, func: &str, artifact: &RecordedArtifact) -> Result<(), String> {
        match self.functions.get(func) {
            Some(bound) if bound.id != artifact.id => Err(format!(
                "function {func:?} is bound to image {} and cannot rebind to {}",
                bound.id, artifact.id
            )),
            Some(_) => Ok(()),
            None => {
                self.functions.insert(func.to_owned(), artifact.clone());
                Ok(())
            }
        }
    }

    /// Identity check for an already-loaded function against the image a
    /// caller asks for, without hashing when the bytes are the bound ones.
    pub fn check_function_image(&self, func: &str, image: &[u8]) -> Result<(), String> {
        match self.functions.get(func) {
            Some(bound) if !bound.holds(image) => Err(format!(
                "function {func:?} is bound to image {} and a different image was requested",
                bound.id
            )),
            _ => Ok(()),
        }
    }

    pub fn module(&self, module: &str) -> Option<&RecordedArtifact> {
        self.modules.get(module)
    }

    pub fn function(&self, func: &str) -> Option<&RecordedArtifact> {
        self.functions.get(func)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static IMAGE_A: &[u8] = b"\x7fELF image a";
    static IMAGE_B: &[u8] = b"\x7fELF image b";

    fn native(module: &'static str, image: &'static [u8]) -> CodeObjectArtifact {
        CodeObjectArtifact::native_embedded(module, image.into(), None)
    }

    #[test]
    fn identity_is_sha256_of_the_exact_image() {
        let artifact = native("m", IMAGE_A);
        assert_eq!(artifact.id(), CodeObjectId(Sha256::digest(IMAGE_A).into()));
        assert_eq!(
            artifact.image().as_bytes().as_ptr(),
            IMAGE_A.as_ptr(),
            "static image is not copied"
        );
        assert_eq!(artifact.origin(), &CodeObjectOrigin::NativeEmbedded("m"));
        let json = artifact.provenance_json();
        assert_eq!(json["origin"], "native_embedded");
        assert_eq!(json["module"], "m");
        assert_eq!(json["image_sha256"], artifact.id().to_hex());
    }

    #[test]
    fn same_symbol_different_image_is_rejected() {
        let mut registry = CodeObjectRegistry::default();
        let a = registry.admit_module("m", native("m", IMAGE_A)).unwrap();
        registry.bind_function("k", &a).unwrap();
        let err = registry
            .admit_module("m", native("m", IMAGE_B))
            .unwrap_err();
        assert!(err.contains("cannot rebind"), "{err}");
        let b = registry
            .admit_module("other", native("other", IMAGE_B))
            .unwrap();
        let err = registry.bind_function("k", &b).unwrap_err();
        assert!(err.contains("cannot rebind"), "{err}");
        assert!(registry.check_function_image("k", IMAGE_A).is_ok());
        assert!(registry
            .check_function_image("k", IMAGE_A.to_vec().as_slice())
            .is_ok());
        assert!(registry.check_function_image("k", IMAGE_B).is_err());
        assert_eq!(registry.function("k").unwrap().id(), a.id());
    }

    #[test]
    fn multi_export_aliases_share_one_artifact() {
        let mut registry = CodeObjectRegistry::default();
        let a = registry
            .admit_module("bundle", native("bundle", IMAGE_A))
            .unwrap();
        for symbol in ["k_b1s", "k_qkv_b1s", "k_gate_up_b1s"] {
            registry.bind_function(symbol, &a).unwrap();
        }
        // Re-admitting the same digest (reload) returns the same handle.
        let again = registry
            .admit_module("bundle", native("bundle", IMAGE_A))
            .unwrap();
        assert!(Arc::ptr_eq(&a, &again));
        for symbol in ["k_b1s", "k_qkv_b1s", "k_gate_up_b1s"] {
            assert!(Arc::ptr_eq(registry.function(symbol).unwrap(), &a));
        }
        assert!(registry.function("absent").is_none());
    }

    #[test]
    fn retained_bytes_outlive_the_source_file() {
        let dir = std::env::temp_dir().join(format!("pm-code-object-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("k.hsaco");
        std::fs::write(&path, IMAGE_A).unwrap();
        let bytes: Arc<[u8]> = std::fs::read(&path).unwrap().into();
        let artifact = Arc::new(CodeObjectArtifact::hip_file(&path, bytes));
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(artifact.image().as_bytes(), IMAGE_A);
        assert_eq!(artifact.id(), CodeObjectId::of(IMAGE_A));
        assert!(artifact.radiowave().is_none());
        assert_eq!(artifact.provenance_json()["origin"], "hip_file");
    }

    #[test]
    fn corrupted_manifest_leaves_no_certification() {
        for manifest in ["", "{", r#"{"compiler":"hipcc"}"#] {
            let artifact = CodeObjectArtifact::native_embedded("m", IMAGE_A.into(), Some(manifest));
            assert!(artifact.radiowave().is_none(), "{manifest:?}");
        }
    }
}
