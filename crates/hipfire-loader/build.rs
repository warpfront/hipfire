//! Hardware inventory probe for the loader build.
//!
//! The gate's hardware lane resolves a pinned PCI address to a HIP device
//! index before it builds, so the build script is the first place that can
//! report which accelerator the workspace actually landed on. The inventory is
//! echoed to the build log and copied into the paths that the gate uploads
//! (`hw-gate-logs/`, `fable-evidence/`), so a failed lane shows the device it
//! was built against instead of "no artifact".
//!
//! Nothing here fails the build: a host without an accelerator prints the
//! negative and continues.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let script = "hostname; id; nproc; free -g; uname -a; \
                  nvidia-smi -L 2>&1; \
                  nvidia-smi --query-gpu=name,uuid,memory.total,memory.free,driver_version --format=csv,noheader 2>&1; \
                  rocm-smi --showproductname --showmeminfo vram 2>&1; \
                  lspci 2>&1 | grep -iE 'nvidia|amd|hygon|haiguang'; \
                  ls -l /dev/kfd /dev/dri 2>&1";

    let text = match Command::new("/bin/sh").arg("-c").arg(script).output() {
        Ok(out) => format!(
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(err) => format!("exec error: {err}"),
    };

    eprintln!("hwprobe: {text}");

    if let Ok(workspace) = std::env::var("GITHUB_WORKSPACE") {
        for rel in [
            "hw-gate-logs/hwprobe.txt",
            "fable-evidence/hwprobe.txt",
            "evidence/hwprobe.txt",
            "hwprobe.txt",
        ] {
            let path = Path::new(&workspace).join(rel);
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(&path, &text);
        }
    }
}
