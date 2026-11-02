// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! `aco-export reference`: original-RADV Vulkan dispatch of the exported SPIR-V on the exact PCI
//! device, producing the `aco-vulkan-ref-v1` receipt consumed by `pm_aco_oracle` (rdna-compute
//! example). The dispatch compiles through the same patched RADV build and the resulting ACO code
//! must be byte-identical to the artifact's `shader.bin`, otherwise no receipt is written.

use crate::export::*;
use crate::mesa;
use crate::util::*;
use serde_json::{json, Map, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const GUARD: u64 = 4096;

fn verify_artifact_files(dir: &Path, art: &Value) -> Res<()> {
    let files = jget(art, "files")?
        .as_object()
        .ok_or("artifact.files must be an object")?;
    for (rel, v) in files {
        let got = sha256_file(&dir.join(rel))?;
        if Some(got.as_str()) != v["sha256"].as_str() {
            return Err(format!(
                "artifact file {rel} sha256 {got} != manifest {}",
                v["sha256"]
            ));
        }
    }
    for must in [
        "shader.spv",
        "shader.bin",
        "shader.abi.json",
        "shader.rawspec.json",
        "shader.aco.dump",
    ] {
        if !files.contains_key(must) {
            return Err(format!("artifact.files lacks {must}"));
        }
    }
    Ok(())
}

fn le_u32(hex: &str, off: usize) -> Res<u32> {
    let b: Vec<u8> = (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(err))
        .collect::<Res<Vec<u8>>>()?;
    if b.len() < off + 4 {
        return Err("kernarg blob too short".into());
    }
    Ok(u32::from_le_bytes(b[off..off + 4].try_into().unwrap()))
}

pub fn cmd_reference(a: &Args) -> Res<()> {
    let spec_path = a
        .opt("spec")
        .map(PathBuf::from)
        .unwrap_or_else(default_spec);
    let spec = read_json(&spec_path)?;
    let art_dir = a.path("artifact")?;
    let art = read_json(&art_dir.join("artifact.json"))?;
    verify_artifact_files(&art_dir, &art)?;
    let kname = jstr(&art, "kernel")?;
    let k = spec_kernel(&spec, kname)?;
    let prep_dir = a.path("prepare-dir")?;
    let cap_dir = a.path("capture")?;
    let out = a.path("out")?;
    let bdf = a.req("pci-bdf")?.to_owned();
    let host = a.path("host")?;
    let mesa_src = a.path("mesa-src")?;
    let mesa_build = a.path("mesa-build")?;
    if out.exists() && fs::read_dir(&out).map_err(err)?.next().is_some() {
        return Err(format!("{} exists and is not empty", out.display()));
    }
    fs::create_dir_all(&out).map_err(err)?;

    let pin = mesa::load_pin()?;
    mesa::verify_tree(&mesa_src, &pin)?;
    let drv = DriverEnv::from_build(&mesa_build, false)?;
    let driver_sha = sha256_file(&drv.driver_lib)?;
    if Some(driver_sha.as_str()) != art["provenance"]["compiler_sha256"].as_str() {
        return Err("RADV driver library differs from the one that produced the artifact".into());
    }
    if sha256_file(&host)? != jstr(&art, "compiler.host.sha256")? {
        return Err("host binary differs from the one that produced the artifact".into());
    }
    let bin_sha = jstr(&art, "binary.sha256")?.to_owned();
    let spv = art_dir.join("shader.spv");

    // fixture identity
    let meta_path = cap_dir.join("meta.json");
    let meta_sha = sha256_file(&meta_path)?;
    let meta = read_json(&meta_path)?;
    if jstr(&meta, "symbol")? != jstr(&art, "symbol")? {
        return Err(format!(
            "capture symbol {} != artifact symbol {}",
            meta["symbol"], art["symbol"]
        ));
    }
    let kernarg_hex = jstr(&meta, "kernarg_hex")?;
    let (cap_m, cap_k) = (le_u32(kernarg_hex, 24)?, le_u32(kernarg_hex, 28)?);

    let prep = read_json(&prep_dir.join("prepare.json"))?;
    if jstr(&prep, "schema")? != "pm-aco-oracle-prepare-v1"
        || ju64(&prep, "guard_bytes")? != GUARD
        || jstr(&prep, "guard_byte")? != "a5"
    {
        return Err("prepare.json schema/guard mismatch".into());
    }
    let refc = jget(k, "reference")?;
    let push_fields = jarr(k, "push.fields")?;
    let buf_for: std::collections::BTreeMap<String, String> = jarr(refc, "pointer_buffers")?
        .iter()
        .map(|e| {
            (
                e["push"].as_str().unwrap_or_default().to_owned(),
                e["buffer"].as_str().unwrap_or_default().to_owned(),
            )
        })
        .collect();
    let local: Vec<String> = jarr(k, "local_size")?
        .iter()
        .map(|v| v.to_string())
        .collect();

    let mut fixture_bucket: Option<Value> = None;
    let mut derived = Map::new();
    let mut identity0: Option<Value> = None;
    let mut runs = Vec::new();
    for shape in jarr(&prep, "shapes")? {
        let scope = jstr(shape, "vulkan_scope")?;
        let name = jstr(shape, "shape")?;
        let derived_key = scope.strip_prefix("derived.");
        let is_fixture = scope.starts_with("fixture");
        if !is_fixture && derived_key.is_none() {
            continue;
        }
        let m = ju64(shape, "M")?;
        let kk = ju64(shape, "K")?;
        let wg: Vec<u64> = jarr(shape, "workgroups")?
            .iter()
            .filter_map(Value::as_u64)
            .collect();
        if wg.len() != 3 {
            return Err("workgroups must have 3 entries".into());
        }
        if is_fixture && (m != cap_m as u64 || kk != cap_k as u64) {
            return Err(format!(
                "fixture shape M/K {m}/{kk} != capture kernarg {cap_m}/{cap_k}"
            ));
        }
        let sdir = out.join(name);
        fs::create_dir_all(&sdir).map_err(err)?;
        let work = out.join(format!("run-{name}"));
        fs::create_dir_all(&work).map_err(err)?;
        let capture_tmp = work.join("capture-tmp");
        fs::create_dir_all(&capture_tmp).map_err(err)?;

        let mut argv: Vec<String> = vec![
            "dispatch".into(),
            "--spv".into(),
            spv.display().to_string(),
            "--push-size".into(),
            ju64(k, "push.size")?.to_string(),
            "--local".into(),
            local.join(","),
            "--subgroup".into(),
            ju64(k, "subgroup.required")?.to_string(),
            "--stats-out".into(),
            work.join("host-stats.txt").display().to_string(),
            "--identity-out".into(),
            work.join("identity.json").display().to_string(),
            "--pci-bdf".into(),
            bdf.clone(),
            "--groups".into(),
            wg.iter().map(u64::to_string).collect::<Vec<_>>().join(","),
            "--out-dir".into(),
            sdir.display().to_string(),
        ];
        let files = jget(shape, "files")?.as_object().ok_or("shape.files")?;
        let mut buffers = Map::new();
        for (bufname, push_name_unused) in buf_for.values().map(|b| (b.clone(), ())) {
            let _ = push_name_unused;
            let pre_name = format!("{bufname}.pre.framed.bin");
            let finfo = files
                .get(&pre_name)
                .ok_or_else(|| format!("prepare.json shape {name} lacks {pre_name}"))?;
            let pre_path = prep_dir.join(name).join(&pre_name);
            let pre_sha = sha256_file(&pre_path)?;
            if Some(pre_sha.as_str()) != finfo["sha256"].as_str() {
                return Err(format!(
                    "{} sha256 differs from prepare.json",
                    pre_path.display()
                ));
            }
            if ju64(finfo, "payload_offset")? != GUARD {
                return Err("payload_offset != guard".into());
            }
            argv.push("--buffer".into());
            argv.push(format!("{bufname}={}:{GUARD}", pre_path.display()));
            buffers.insert(
                bufname.clone(),
                json!({"pre_framed_sha256": pre_sha, "bytes": finfo["bytes"]}),
            );
        }
        // push block in field order
        let mut items: Vec<String> = Vec::new();
        let mut fields: Vec<&Value> = push_fields.iter().collect();
        fields.sort_by_key(|f| f["offset"].as_u64().unwrap_or(0));
        let mut cursor = 0u64;
        for f in fields {
            let off = ju64(f, "offset")?;
            if off != cursor {
                return Err(format!("push fields leave a gap at byte {cursor}"));
            }
            let fname = jstr(f, "name")?;
            match jstr(f, "kind")? {
                "u64_ptr" => items.push(format!(
                    "u64:buf:{}",
                    buf_for
                        .get(fname)
                        .ok_or_else(|| format!("no reference buffer for push field {fname}"))?
                )),
                "i32" => {
                    let v = match jstr(f, "reference_value")? {
                        "M" => m,
                        "K" => kk,
                        other => return Err(format!("unknown reference_value {other}")),
                    };
                    items.push(format!("i32:{v}"));
                }
                other => {
                    return Err(format!(
                        "reference mode cannot synthesize push kind {other}"
                    ))
                }
            }
            cursor += ju64(f, "size")?;
        }
        if cursor != ju64(k, "push.size")? {
            return Err("push fields do not cover the push block".into());
        }
        argv.push("--push".into());
        argv.push(items.join(","));

        let mut cmd = Command::new(&host);
        cmd.args(&argv);
        let env = drv.apply(&mut cmd, &capture_tmp);
        let o = cmd.output().map_err(|e| format!("spawn host: {e}"))?;
        write_file(&work.join("host.stderr.txt"), &o.stderr)?;
        write_file(&work.join("host.stdout.txt"), &o.stdout)?;
        if !o.status.success() || !String::from_utf8_lossy(&o.stdout).contains("dispatched") {
            return Err(format!(
                "vulkan dispatch of shape {name} failed ({}):\n{}",
                o.status,
                String::from_utf8_lossy(&o.stderr)
            ));
        }
        // the executed shader must be the exported binary
        let dirs: Vec<PathBuf> = fs::read_dir(&capture_tmp)
            .map_err(err)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        if dirs.len() != 1 {
            return Err(format!(
                "shape {name}: expected exactly one compute capture, found {}",
                dirs.len()
            ));
        }
        let ran_sha = sha256_file(&dirs[0].join("shader.bin"))?;
        if ran_sha != bin_sha {
            return Err(format!("shape {name}: the Vulkan-executed ACO binary {ran_sha} != artifact shader.bin {bin_sha}; refusing a receipt"));
        }
        let ident = read_json(&work.join("identity.json"))?;
        if jstr(&ident, "selection")? != "pci_bdf"
            || !jstr(&ident, "pci_bdf")?.eq_ignore_ascii_case(&bdf)
        {
            return Err("identity.json does not carry the requested PCI BDF".into());
        }
        match &identity0 {
            None => identity0 = Some(ident.clone()),
            Some(i0) if i0 != &ident => return Err("device identity changed between shapes".into()),
            _ => {}
        }

        for (bufname, b) in buffers.iter_mut() {
            let post = sdir.join(format!("{bufname}.post.bin"));
            let bytes = fs::read(&post).map_err(|e| format!("{}: {e}", post.display()))?;
            if bytes.len() as u64 != b["bytes"].as_u64().unwrap_or(0) {
                return Err(format!(
                    "{} length {} != framed pre length {}",
                    post.display(),
                    bytes.len(),
                    b["bytes"]
                ));
            }
            b["post_framed_file"] = json!(format!("{name}/{bufname}.post.bin"));
            b["post_framed_sha256"] = json!(sha256_hex(&bytes));
            b.as_object_mut().unwrap().remove("bytes");
        }
        let bucket = json!({"config": {"M": m, "K": kk, "workgroups": wg, "local_size": k["local_size"]}, "buffers": Value::Object(buffers)});
        runs.push(json!({"shape": name, "scope": scope, "argv": argv, "environment": env}));
        if is_fixture {
            fixture_bucket = Some(bucket);
        } else {
            derived.insert(derived_key.unwrap().to_owned(), bucket);
        }
    }
    let fb = fixture_bucket.ok_or("prepare.json has no fixture shape")?;
    let ident = identity0.unwrap();
    let mesa_version = jstr(&ident, "driver_info")?.to_owned();
    let tag_version = jstr(&pin, "tag")?.trim_start_matches("mesa-").to_owned();
    if !mesa_version.contains(&tag_version)
        && !jstr(&ident, "driver_name")?.to_lowercase().contains("radv")
    {
        return Err(format!(
            "driver identity `{mesa_version}` does not show the pinned Mesa {tag_version}"
        ));
    }

    let receipt = json!({
        "schema": "aco-vulkan-ref-v1",
        "symbol": jstr(&art, "symbol")?,
        "spirv": {"sha256": jstr(&art, "spirv.sha256")?},
        "aco_binary": {
            "sha256": bin_sha, "sidecar_sha256": jstr(&art, "abi.sha256")?,
            "executable_sha256": jstr(&art, "binary.executable.sha256")?,
            "executable_offset": jget(&art, "binary.executable.offset")?, "executable_len": jget(&art, "binary.executable.len")?,
            "rawspec_sha256": jstr(&art, "rawspec.sha256")?,
            "artifact_json_sha256": sha256_file(&art_dir.join("artifact.json"))?,
        },
        "config": {
            "wave_size": jget(&art, "target.wave")?, "local_size": k["local_size"], "workgroups": fb["config"]["workgroups"],
            "M": fb["config"]["M"], "K": fb["config"]["K"],
            "specialization_sha256": jstr(&art, "pipeline.specialization_sha256")?, "layout_sha256": jstr(&art, "pipeline.layout_sha256")?,
        },
        "fixture": {"meta_sha256": meta_sha, "capture_dir": cap_dir.display().to_string()},
        "driver": {
            "mesa_version": mesa_version, "device_name": ident["device_name"], "vendor_id": ident["vendor_id"], "device_id": ident["device_id"],
            "driver_id": "VK_DRIVER_ID_MESA_RADV", "driver_id_value": ident["driver_id"], "driver_name": ident["driver_name"],
            "pci_bdf": ident["pci_bdf"], "device_uuid": ident["device_uuid"], "driver_uuid": ident["driver_uuid"],
            "api_version": ident["api_version"], "driver_version": ident["driver_version"], "selection": "pci_bdf",
            "driver_library_sha256": driver_sha, "mesa_commit": jstr(&pin, "commit")?, "mesa_patch_sha256": jstr(&art, "provenance.mesa_patch_sha256")?,
        },
        "framing": {"guard_bytes": GUARD, "guard_byte": "a5"},
        "buffers": fb["buffers"],
        "derived": Value::Object(derived),
        "reproduction": {"runs": runs, "host_sha256": sha256_file(&host)?},
    });
    write_file(&out.join("receipt.json"), &canon(&receipt))?;
    println!("receipt: {}", out.join("receipt.json").display());
    Ok(())
}
