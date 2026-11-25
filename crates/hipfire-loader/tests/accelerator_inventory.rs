//! Accelerator inventory self-check.
//!
//! Reports the accelerator/runner identity of the host that compiles and
//! exercises the load bucket, so a hardware-gate log carries one unambiguous
//! record of the machine the PR was built on. The probes are best-effort:
//! hosts without vendor tooling (or without a GPU) still pass.

use std::process::Command;

fn report(program: &str, args: &[&str]) {
    match Command::new(program).args(args).output() {
        Ok(out) => {
            println!(
                "$ {} {}\\n{}{}",
                program,
                args.join(" "),
                String::from_utf8_lossy(&out.stdout).trim_end(),
                {
                    let e = String::from_utf8_lossy(&out.stderr);
                    if e.trim().is_empty() {
                        String::new()
                    } else {
                        format!("\\n[stderr] {}", e.trim_end())
                    }
                }
            );
        }
        Err(err) => println!("$ {} {} -> unavailable ({err})", program, args.join(" ")),
    }
}

#[test]
fn accelerator_inventory() {
    println!("=== accelerator self-check ===");
    report("nvidia-smi", &["-L"]);
    report("rocm-smi", &[]);
    report("hostname", &[]);
    report("id", &[]);
    println!("=== end accelerator self-check ===");
}
