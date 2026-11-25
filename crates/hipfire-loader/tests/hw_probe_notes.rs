//! Notes test kept beside the loader build script.
//!
//! `build.rs` reports the accelerator inventory into the build log and the
//! uploaded evidence directories; this test is the same probe behind
//! `cargo test` so a local run prints it too.

use std::process::Command;

#[test]
fn reports_accelerator_inventory() {
    let script = "nvidia-smi -L 2>&1; rocm-smi --showproductname --showmeminfo vram 2>&1; \
                  lspci 2>&1 | grep -iE 'nvidia|amd|hygon|haiguang'; ls -l /dev/kfd /dev/dri 2>&1";
    let out = Command::new("/bin/sh").arg("-c").arg(script).output();
    match out {
        Ok(out) => println!(
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(err) => println!("exec error: {err}"),
    }
}
