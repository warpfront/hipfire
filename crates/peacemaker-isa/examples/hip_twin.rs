// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hip_twin --in X.co --arch gfx1100 --out OUT.hxaco [--report R.json]
//!   --kernel SYM [--abi-ref INC.co --abi-ref-symbol S] [--kernel SYM2 [--abi-ref ... --abi-ref-symbol ...]]`
//! Seals a hipcc object as an `.hxaco` (+ `.hxaco.sha256`) and prints the report JSON.
//! `--kernel` may repeat (one hxaco holding several twins); `--abi-ref`/`--abi-ref-symbol`
//! attach to the preceding `--kernel`. One kernel prints its report object, several print an
//! array of reports in `--kernel` order.

use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{self, KernelSpec};

struct Kern {
    symbol: String,
    aref: Option<String>,
    asym: Option<String>,
}

const USAGE: &str = "usage: hip_twin --in X.co --arch gfx1100 --out OUT.hxaco [--report R.json] \
--kernel SYM [--abi-ref INC.co --abi-ref-symbol S] [--kernel SYM2 ...]";

fn run() -> Result<(), String> {
    let (mut inp, mut arch, mut out, mut rep) = (None, None, None, None);
    let mut kernels: Vec<Kern> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--in" => inp = Some(val()?),
            "--arch" => arch = Some(val()?),
            "--out" => out = Some(val()?),
            "--report" => rep = Some(val()?),
            "--kernel" => kernels.push(Kern {
                symbol: val()?,
                aref: None,
                asym: None,
            }),
            "--abi-ref" | "--abi-ref-symbol" => {
                let v = val()?;
                let k = kernels
                    .last_mut()
                    .ok_or(format!("{a} must follow a --kernel"))?;
                let slot = if a == "--abi-ref" {
                    &mut k.aref
                } else {
                    &mut k.asym
                };
                if slot.replace(v).is_some() {
                    return Err(format!("{a} given twice for kernel {}", k.symbol));
                }
            }
            _ => return Err(format!("unknown argument {a}")),
        }
    }
    let (inp, arch, out) = (
        inp.ok_or(USAGE)?,
        arch.ok_or(USAGE)?.parse::<Arch>()?,
        out.ok_or(USAGE)?,
    );
    if kernels.is_empty() {
        return Err(USAGE.into());
    }
    let bytes = std::fs::read(&inp).map_err(|e| format!("{inp}: {e}"))?;
    let mut refs: Vec<Option<(Vec<u8>, String)>> = Vec::new();
    for k in &kernels {
        refs.push(match (&k.aref, &k.asym) {
            (Some(p), Some(s)) => Some((
                std::fs::read(p).map_err(|e| format!("{p}: {e}"))?,
                s.clone(),
            )),
            (None, None) => None,
            _ => {
                return Err(format!(
                    "kernel {}: --abi-ref and --abi-ref-symbol go together",
                    k.symbol
                ))
            }
        });
    }
    let specs: Vec<KernelSpec<'_>> = kernels
        .iter()
        .zip(&refs)
        .map(|(k, r)| KernelSpec {
            symbol: &k.symbol,
            abi_ref: r.as_ref().map(|(b, s)| (b.as_slice(), s.as_str())),
        })
        .collect();
    let sealed = hip_twin::seal_many(&bytes, arch, &specs)?;
    let symbols: Vec<&str> = kernels.iter().map(|k| k.symbol.as_str()).collect();
    hip_twin::verify_hxaco_many(&sealed[0].hxaco, arch, &symbols)?;
    std::fs::write(&out, &sealed[0].hxaco).map_err(|e| format!("{out}: {e}"))?;
    std::fs::write(format!("{out}.sha256"), &sealed[0].hxaco_sha256).map_err(|e| e.to_string())?;
    let json = if sealed.len() == 1 {
        serde_json::to_string_pretty(&sealed[0].report)
    } else {
        serde_json::to_string_pretty(&sealed.iter().map(|s| &s.report).collect::<Vec<_>>())
    }
    .unwrap();
    if let Some(r) = rep {
        std::fs::write(&r, &json).map_err(|e| format!("{r}: {e}"))?;
    }
    println!("{json}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("hip_twin: {e}");
        std::process::exit(1);
    }
}
