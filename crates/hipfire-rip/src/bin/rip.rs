//! `rip`: compile a `.rip` kernel script to AMDGCN assembly without cargo.
//!
//! ```text
//! rip [--certify] SOURCE.rip ARCH VARIANT OUTPUT.s
//! rip [--certify] --module SOURCE.rip ARCH VARIANT[,VARIANT..] MODULE_NAME OUTPUT.s
//! ```
//!
//! `ARCH` is `gfx1151`, `gfx1201`, `gfx1100` or `both` (gfx1151 and gfx1201;
//! `OUTPUT` then gets the arch before its extension: `out.gfx1151.s`).
//! `--certify` assembles and links the emitted code object into
//! `$RIP_CERT_DIR` (default: a temporary directory), then runs the
//! existing passes on it: wait-ledger replay and M7 (byte-exact lift, no
//! obligations). The bundler's host target identity is the toolchain default;
//! to reproduce a committed `.hxaco` recipe configure the library's
//! `Toolchain` instead.

use hipfire_isa::Arch;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn usage() -> ExitCode {
    eprintln!("usage: rip [--certify] SOURCE.rip ARCH VARIANT OUTPUT.s\n       rip [--certify] --module SOURCE.rip ARCH VARIANT[,VARIANT..] MODULE_NAME OUTPUT.s");
    ExitCode::from(2)
}

fn arches(a: &str) -> Result<Vec<Arch>, String> {
    if a == "both" {
        return Ok(vec![Arch::Gfx1151, Arch::Gfx1201]);
    }
    a.parse::<Arch>().map(|x| vec![x])
}

fn out_path(out: &str, arch: Arch, many: bool) -> PathBuf {
    let p = Path::new(out);
    if !many {
        return p.to_path_buf();
    }
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
    let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("s");
    p.with_file_name(format!("{stem}.{}.{ext}", arch.name()))
}

fn run() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let certify = args.iter().position(|a| a == "--certify").map(|i| args.remove(i)).is_some();
    let module = args.iter().position(|a| a == "--module").map(|i| args.remove(i)).is_some();
    let want = if module { 5 } else { 4 };
    if args.len() != want {
        return Err("bad arguments".into());
    }
    let source = std::fs::read_to_string(&args[0]).map_err(|e| format!("{}: {e}", args[0]))?;
    let arches = arches(&args[1])?;
    let many = arches.len() > 1;
    for arch in arches {
        let (emitted, text, name) = if module {
            let variants: Vec<&str> = args[2].split(',').collect();
            let (emitted, text, _) = hipfire_rip::module_named(&source, &args[0], arch, &variants, &args[3])?;
            (emitted, text, args[3].clone())
        } else {
            let e = hipfire_rip::compile_named(&source, &args[0], arch, &args[2])?;
            let text = e.s_text.clone();
            let name = hipfire_rip::symbol_of(&e)?;
            (vec![e], text, name)
        };
        let out = out_path(&args[want - 1], arch, many);
        std::fs::write(&out, &text).map_err(|e| format!("{}: {e}", out.display()))?;
        eprintln!("{}: {} kernel(s) for {} -> {}", args[0], emitted.len(), arch.name(), out.display());
        if certify {
            #[cfg(feature = "certify")]
            {
                let dir = std::env::var_os("RIP_CERT_DIR").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join(format!("rip-cert-{}-{}", std::process::id(), arch.name())));
                for c in hipfire_rip::certify(&emitted, &text, arch, &name, &dir)? {
                    eprintln!("certified {} ({}): lift {}, obligations {}", c.symbol, arch.name(), c.m7["lift"], c.m7["obligations"]);
                }
            }
            #[cfg(not(feature = "certify"))]
            {
                let _ = &name;
                return Err("built without the `certify` feature".into());
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e == "bad arguments" => usage(),
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
