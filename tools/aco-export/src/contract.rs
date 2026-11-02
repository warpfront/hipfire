// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! `aco-export contract`: derive the pilot `pm-aco-hip-contract-v1` (consumed by
//! `peacemaker_isa::aco_hsa::AcoHipContract::from_json`) from the captured sidecar/rawspec and the real
//! capture fixture. Nothing is typed in by hand: pointer bindings come from the sidecar's proven
//! inline-user-SGPR roles, scalars from the push block, shape from the capture meta, FP/resources
//! from the rawspec, hashes from the artifact files. An SGPR that the body mentions but that has no
//! HIP source (ring offsets, tg_size, ...) is a rejection.

use crate::export::spec_kernel;
use crate::util::*;
use serde_json::{json, Value};
use std::fs;
use std::path::PathBuf;

/// Evaluate `A*B/C*...` left to right over u128 with the identifiers M and K.
fn eval_extent(expr: &str, m: u64, k: u64) -> Res<u64> {
    let mut acc: u128 = 0;
    let mut op = b'*';
    let mut first = true;
    let mut tok = String::new();
    let mut chars: Vec<char> = expr.chars().filter(|c| !c.is_whitespace()).collect();
    chars.push('\0');
    for c in chars {
        if c == '*' || c == '/' || c == '\0' {
            let v: u128 = match tok.as_str() {
                "M" => m as u128,
                "K" => k as u128,
                t => t
                    .parse()
                    .map_err(|_| format!("bad extent token `{t}` in `{expr}`"))?,
            };
            if first {
                acc = v;
                first = false;
            } else if op == b'*' {
                acc *= v;
            } else {
                if v == 0 || acc % v != 0 {
                    return Err(format!(
                        "extent `{expr}` is not an exact integer division for M={m} K={k}"
                    ));
                }
                acc /= v;
            }
            op = if c == '/' { b'/' } else { b'*' };
            tok.clear();
        } else {
            tok.push(c);
        }
    }
    u64::try_from(acc).map_err(|_| "extent overflows u64".to_owned())
}

fn le_u32(hex: &str, off: usize) -> Res<u64> {
    let b: Vec<u8> = (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(err))
        .collect::<Res<Vec<u8>>>()?;
    if b.len() < off + 4 {
        return Err("kernarg blob too short".into());
    }
    Ok(u32::from_le_bytes(b[off..off + 4].try_into().unwrap()) as u64)
}

pub fn cmd_contract(a: &Args) -> Res<()> {
    let spec_path = a
        .opt("spec")
        .map(PathBuf::from)
        .unwrap_or_else(default_spec);
    let spec = read_json(&spec_path)?;
    let dir = a.path("artifact")?;
    let art_path = dir.join("artifact.json");
    let art = read_json(&art_path)?;
    let k = spec_kernel(&spec, jstr(&art, "kernel")?)?;
    let hip = jget(k, "hip")?;
    let rawspec = read_json(&dir.join("shader.rawspec.json"))?;
    let abi = read_json(&dir.join("shader.abi.json"))?;
    // artifact files must still hash to the manifest
    for (rel, v) in jget(&art, "files")?.as_object().ok_or("files")? {
        let got = sha256_file(&dir.join(rel))?;
        if Some(got.as_str()) != v["sha256"].as_str() {
            return Err(format!("artifact file {rel} differs from artifact.json"));
        }
    }

    let meta = read_json(&a.path("capture")?.join("meta.json"))?;
    if jstr(&meta, "symbol")? != jstr(&art, "symbol")? {
        return Err("capture symbol != artifact symbol".into());
    }
    let kernarg_hex = jstr(&meta, "kernarg_hex")?;
    let mval = le_u32(kernarg_hex, 24)?;
    let kval = le_u32(kernarg_hex, 28)?;
    let block: Vec<u64> = jarr(&meta, "block")?
        .iter()
        .filter_map(Value::as_u64)
        .collect();
    let grid: Vec<u64> = jarr(&meta, "grid")?
        .iter()
        .filter_map(Value::as_u64)
        .collect();
    let local: Vec<u64> = jarr(k, "local_size")?
        .iter()
        .filter_map(Value::as_u64)
        .collect();
    if block != local {
        return Err(format!(
            "capture block {block:?} != compiled local_size {local:?}"
        ));
    }
    if ju64(&meta, "dynamic_lds")? != 0 {
        return Err("capture used dynamic LDS; the compiled shader has none".into());
    }
    if grid != vec![mval, 1, 1] {
        return Err(format!(
            "capture grid {grid:?} is not the launcher rule (M,1,1) for M={mval}"
        ));
    }

    // HIP args (kernarg ABI of the incumbent)
    let mut args = Vec::new();
    let mut names: Vec<String> = Vec::new();
    for ha in jarr(hip, "args")? {
        let name = jstr(ha, "name")?.to_owned();
        let kind = jstr(ha, "kind")?;
        if kind == "global_pointer" {
            let access = jstr(ha, "access")?;
            let extent = eval_extent(jstr(ha, "extent")?, mval, kval)?;
            if let Some(file) = meta["files"].get(format!("arg{}.pre.bin", ju64(ha, "offset")?)) {
                if file["bytes"].as_u64() != Some(extent) {
                    return Err(format!(
                        "arg {name}: extent {extent} != captured arg{}.pre.bin bytes {}",
                        ha["offset"], file["bytes"]
                    ));
                }
            }
            args.push(json!({"name": name, "offset": ha["offset"], "size": ha["size"], "kind": kind, "access": access, "extent_bytes": extent}));
        } else if kind == "by_value" {
            args.push(json!({"name": name, "offset": ha["offset"], "size": ha["size"], "kind": kind, "access": null, "extent_bytes": null}));
        } else {
            return Err(format!("unknown hip arg kind {kind}"));
        }
        names.push(name);
    }
    let arg_index = |n: &str| {
        names
            .iter()
            .position(|x| x == n)
            .ok_or_else(|| format!("hip arg `{n}` not declared"))
    };

    // Entry bindings from the sidecar
    let mut bindings = Vec::new();
    for e in jarr(&abi, "user_sgprs")? {
        let role = jstr(e, "role")?;
        let base = ju64(e, "sgpr_index")?;
        let len = ju64(e, "dwords")?;
        let dst = json!({"kind": "sgpr", "base": base, "len": len});
        let live = e["entry_live"]
            .as_bool()
            .ok_or("sidecar entry lacks entry_live (re-export with the typed-liveness exporter)")?;
        let source = match role {
            "bda_ptr" => {
                if len != 2 {
                    return Err("bda_ptr must be 2 dwords".into());
                }
                json!({"kind": "kernarg_pointer", "arg": arg_index(jstr(e, "hip_arg")?)?})
            }
            "push_u32" => json!({"kind": "kernarg_scalar", "arg": arg_index(jstr(e, "hip_arg")?)?}),
            "wgid_x" | "wgid_y" | "wgid_z" => json!({"kind": "workgroup_id", "axis": e["axis"]}),
            "ring" | "other" if !live => continue, // not live-in per ACO typed liveness: no entry fact, no binding
            r => return Err(format!("s{base} role `{r}` is LIVE-IN at the entry (typed ACO liveness) but has no HIP source")),
        };
        bindings.push(json!({"dst": dst, "source": source}));
    }

    let mut fp = rawspec["fp"].clone();
    if fp.get("wg_round_robin").is_none() {
        return Err("rawspec fp lacks wg_round_robin (re-export)".into());
    }
    let _ = fp.as_object_mut();
    let scalars: Vec<Value> = jarr(hip, "scalars")?
        .iter()
        .map(|s| -> Res<Value> {
            let name = jstr(s, "arg")?;
            let v = match jstr(s, "value")? {
                "M" => mval,
                "K" => kval,
                o => return Err(format!("unknown scalar source {o}")),
            };
            Ok(json!({"arg": arg_index(name)?, "value": v}))
        })
        .collect::<Res<Vec<_>>>()?;
    let odd = match a.opt("odd-tail-evidence-sha256") {
        Some(h) => {
            if h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("--odd-tail-evidence-sha256 must be 64 hex digits".into());
            }
            json!(h)
        }
        None => Value::Null,
    };

    let contract = json!({
        "schema": "pm-aco-hip-contract-v1",
        "target": rawspec["target"],
        "raw_symbol": rawspec["symbol"],
        "hip_symbol": jstr(hip, "symbol")?,
        "wave": rawspec["wave"],
        "kernarg_size": jget(hip, "kernarg_size")?,
        "args": args,
        "bindings": bindings,
        "fp": fp,
        "resources": rawspec["resources"],
        "dynamic_lds_max": 0,
        "shape": {"block": block, "grid": grid, "scalars": scalars, "odd_tail_evidence_sha256": odd},
        "aperture": "global64",
        "artifact_sha256": sha256_file(&art_path)?,
        "rawspec_sha256": sha256_file(&dir.join("shader.rawspec.json"))?,
        "spirv_sha256": sha256_file(&dir.join("shader.spv"))?,
        "raw_binary_sha256": jstr(&art, "binary.sha256")?,
        "raw_exec_sha256": jstr(&art, "binary.executable.sha256")?,
        "sidecar_sha256": sha256_file(&dir.join("shader.abi.json"))?,
    });
    let out = a.path("out")?;
    write_file(&out, &canon(&contract))?;
    println!("contract: {} sha256 {}", out.display(), sha256_file(&out)?);
    let _ = fs::metadata(&out);
    Ok(())
}
