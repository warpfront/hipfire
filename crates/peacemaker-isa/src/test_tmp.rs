//! Scratch code objects for unit tests. Written under the workspace `target/`
//! (never the source tree's `kernels/`, which `tests/ledger_cfg_pins.rs` scans).
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

/// A temporary file removed on drop (including on panic).
pub(crate) struct TmpObject(PathBuf);

impl TmpObject {
    pub(crate) fn write(stem: &str, bytes: impl AsRef<[u8]>) -> Self {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/pm-test-tmp");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!(
            "{stem}.{}.{}.co",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TmpObject {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
