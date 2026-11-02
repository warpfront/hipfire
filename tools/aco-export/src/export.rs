// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! `aco-export export`: GLSL -> SPIR-V (glslang, validated) -> real RADV (SPIR-V -> NIR -> ACO)
//! through the Vulkan host -> pm-aco-artifact-v1.

use crate::abi;
use crate::mesa;
use crate::util::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const ARTIFACT_SCHEMA: &str = "pm-aco-artifact-v1";

pub struct DriverEnv {
    pub icd_json: PathBuf,
    pub driver_lib: PathBuf,
    pub shim_lib: Option<PathBuf>,
    pub shim_gpu_id: String,
}

impl DriverEnv {
    pub fn from_build(build: &Path, shim: bool) -> Res<Self> {
        let icd_json = build.join("src/amd/vulkan/radeon_devenv_icd.x86_64.json");
        let driver_lib = build.join("src/amd/vulkan/libvulkan_radeon.so");
        for p in [&icd_json, &driver_lib] {
            if !p.is_file() {
                return Err(format!(
                    "{} missing (run `aco-export build-mesa` first)",
                    p.display()
                ));
            }
        }
        let shim_lib = if shim {
            let p = build.join("src/amd/drm-shim/libamdgpu_noop_drm_shim.so");
            if !p.is_file() {
                return Err(format!("{} missing", p.display()));
            }
            Some(p)
        } else {
            None
        };
        Ok(Self {
            icd_json,
            driver_lib,
            shim_lib,
            shim_gpu_id: "gfx1201".into(),
        })
    }

    /// The complete, explicit environment of every host run. The inherited environment is dropped so
    /// that stray RADV_*/MESA_*/VK_* settings cannot change codegen; what remains is recorded.
    pub fn apply(&self, cmd: &mut Command, capture_dir: &Path) -> BTreeMap<String, String> {
        let mut env: BTreeMap<String, String> = BTreeMap::new();
        for k in ["PATH", "HOME", "LD_LIBRARY_PATH"] {
            if let Ok(v) = std::env::var(k) {
                env.insert(k.into(), v);
            }
        }
        env.insert(
            "VK_DRIVER_FILES".into(),
            self.icd_json.display().to_string(),
        );
        env.insert(
            "VK_ICD_FILENAMES".into(),
            self.icd_json.display().to_string(),
        );
        env.insert("VK_LOADER_LAYERS_DISABLE".into(), "~all~".into());
        env.insert(
            "RADV_ACO_EXPORT_DIR".into(),
            capture_dir.display().to_string(),
        );
        env.insert("RADV_DEBUG".into(), "nocache".into());
        env.insert("MESA_SHADER_CACHE_DISABLE".into(), "true".into());
        if let Some(s) = &self.shim_lib {
            env.insert("LD_PRELOAD".into(), s.display().to_string());
            env.insert("AMDGPU_GPU_ID".into(), self.shim_gpu_id.clone());
        }
        cmd.env_clear();
        for (k, v) in &env {
            cmd.env(k, v);
        }
        env
    }
}

pub fn find_llvm_mc(a: &Args) -> Res<PathBuf> {
    if let Some(p) = a.opt("llvm-mc") {
        return Ok(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("ACO_LLVM_MC") {
        return Ok(PathBuf::from(p));
    }
    for p in [
        "/opt/rocm/core/lib/llvm/bin/llvm-mc",
        "/opt/rocm/llvm/bin/llvm-mc",
        "/opt/rocm/core-10.0/lib/llvm/bin/llvm-mc",
    ] {
        if Path::new(p).is_file() {
            return Ok(PathBuf::from(p));
        }
    }
    Err("no llvm-mc found (pass --llvm-mc; the pinned /opt/rocm/core-10.0 path is absent on this host)".into())
}

/// Disassemble exactly the executable range with ROCm llvm-mc (Mesa's own print_asm is unusable on
/// gfx1201 with its LLVM 18).
fn disassemble(llvm_mc: &Path, arch: &str, wave: u64, exec: &[u8]) -> Res<String> {
    use std::io::Write;
    let mut text = String::new();
    for (i, b) in exec.iter().enumerate() {
        if i > 0 {
            text.push(',');
        }
        text.push_str(&format!("0x{b:02x}"));
    }
    text.push('\n');
    let attr = if wave == 32 {
        "+wavefrontsize32,-wavefrontsize64"
    } else {
        "-wavefrontsize32,+wavefrontsize64"
    };
    let mut child = Command::new(llvm_mc)
        .args([
            "--disassemble",
            "--triple=amdgcn-amd-amdhsa",
            &format!("--mcpu={arch}"),
            &format!("-mattr={attr}"),
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn llvm-mc: {e}"))?;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(text.as_bytes())
        .map_err(err)?;
    let out = child.wait_with_output().map_err(err)?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success()
        || stderr.contains("invalid instruction encoding")
        || stderr.contains("error:")
    {
        return Err(format!(
            "llvm-mc could not disassemble every executable word:\n{stderr}"
        ));
    }
    let asm = String::from_utf8_lossy(&out.stdout);
    let body: Vec<&str> = asm
        .lines()
        .map(str::trim_end)
        .filter(|l| {
            !l.trim().is_empty()
                && !l.trim_start().starts_with(".text")
                && !l.trim_start().starts_with("//")
        })
        .collect();
    Ok(body.join("\n"))
}

fn mask_bit(masks: &[u64], i: u64) -> bool {
    masks
        .get((i / 32) as usize)
        .map(|m| (m >> (i % 32)) & 1 == 1)
        .unwrap_or(false)
}

fn parse_masks(t: &Value, key: &str) -> Res<Vec<u64>> {
    let arr = jarr(t, key)?;
    if arr.len() != 4 {
        return Err(format!("typed_post_ra.{key} must have 4 words"));
    }
    arr.iter()
        .map(|v| {
            u64::from_str_radix(v.as_str().unwrap_or(""), 16)
                .map_err(|e| format!("typed_post_ra.{key}: {e}"))
        })
        .collect()
}

/// Bind register facts that come from the compiler itself: ACO's typed post-RA instructions (every fixed
/// Operand/Definition PhysReg range after all ACO passes, `typed_post_ra` in the capture). Nothing is
/// derived from disassembly text.
///  * a `ring`/`other` entry SGPR that no typed operand reads and no typed definition touches gets NO entry
///    fact (any read would be a rejected undefined read); one the body touches keeps a `defined` fact and
///    fails contract generation (no HIP source).
///  * SGPR counts (gfx10+): the 108 RADV/ACO hardware total is 106 ordinary SGPRs (s0..s105) plus VCC
///    (s106..s107): ac_gpu_info.c:253 `max_sgpr_alloc = 108; /* includes VCC, which can be treated as s[106-107]
///    on GFX10+ */`, aco_ir.h:438-439 `vcc{106}`, `vcc_hi{107}`. ACO's RA max-used (107 when VCC_LO is used)
///    includes VCC. Typed RawResources counts are ORDINARY SGPRs only; hw total / RA count / VCC touched are
///    kept verbatim in config. Nothing is truncated.
fn apply_body_reads(an: &mut abi::Analysis, cap: &Value) -> Res<()> {
    let t = jget(cap, "typed_post_ra")?;
    let reads = parse_masks(t, "sgpr_read_mask")?;
    let defs = parse_masks(t, "sgpr_def_mask")?;
    let live_in = parse_masks(t, "entry_live_mask")?;
    let typed_ordinary = ju64(t, "ordinary_sgprs")?;
    if typed_ordinary > 106 {
        return Err(format!(
            "REJECT: typed ordinary SGPR count {typed_ordinary} > 106"
        ));
    }
    let mut drop_bases: Vec<u64> = Vec::new();
    if let Some(list) = an.abi["user_sgprs"].as_array_mut() {
        for e in list.iter_mut() {
            let base = e["sgpr_index"].as_u64().unwrap_or(u64::MAX);
            let len = e["dwords"].as_u64().unwrap_or(0);
            if base + len > 106 {
                return Err(format!(
                    "REJECT: entry SGPR s{base}+{len} is not ordinary (>=106)"
                ));
            }
            let read = (base..base + len).any(|r| mask_bit(&reads, r));
            let defd = (base..base + len).any(|r| mask_bit(&defs, r));
            let live = (base..base + len).any(|r| mask_bit(&live_in, r));
            e["entry_live"] = json!(live);
            e["body_reads"] = json!(read);
            e["body_writes"] = json!(defd);
            e["entry_live_evidence"] = json!("ACO typed post-RA instructions, backward liveness over the linear CFG (capture.typed_post_ra.entry_live_mask): the entry value is read before any write on some path. body_reads/body_writes are plain operand/definition touches (not liveness).");
            let role = e["role"].as_str().unwrap_or("");
            if (role == "ring" || role == "other") && !live {
                drop_bases.push(base);
                e["entry_fact_emitted"] = json!(false);
            } else {
                e["entry_fact_emitted"] = json!(true);
            }
        }
    }
    an.entry.retain(|f| {
        !(f["loc"]["kind"] == "sgpr"
            && f["value"]["kind"] == "defined"
            && drop_bases.contains(&f["loc"]["base"].as_u64().unwrap_or(u64::MAX)))
    });

    let hw_total = an.config["sgprs_allocated"]
        .as_u64()
        .ok_or("config.sgprs_allocated")?;
    let ra_incl_vcc = an.config["sgprs_actual"]
        .as_u64()
        .ok_or("config.sgprs_actual")?;
    if hw_total != 108 {
        return Err(format!("REJECT: final config.num_sgprs {hw_total} != 108 (gfx10+ fixed total, ac_gpu_info.c:253); SGPR semantics unverified"));
    }
    let user_end = an.abi["user_sgprs"]
        .as_array()
        .map(|l| {
            l.iter()
                .map(|e| e["sgpr_index"].as_u64().unwrap_or(0) + e["dwords"].as_u64().unwrap_or(0))
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0);
    let ordinary_actual = user_end.max(typed_ordinary);
    let vcc = jbool(t, "vcc_touched")?;
    let other_special = jbool(t, "other_special_touched")?;
    an.resources["sgprs_actual"] = json!(ordinary_actual);
    an.resources["sgprs_allocated"] = json!(106);
    for cfg in [&mut an.config, &mut an.abi["config"]] {
        cfg["sgprs_actual"] = json!(ordinary_actual);
        cfg["sgprs_allocated"] = json!(106);
        cfg["sgprs_hw_total"] = json!(hw_total);
        cfg["sgprs_ra_max_used_incl_vcc"] = json!(ra_incl_vcc);
        cfg["sgprs_typed_ordinary"] = json!(typed_ordinary);
        cfg["sgprs_vcc_touched"] = json!(vcc);
        cfg["sgprs_other_special_touched"] = json!(other_special);
        cfg["sgprs_semantics"] = json!("sgprs_actual/allocated are ORDINARY SGPRs (s0..s105): actual = max(end of the entry-slot SGPRs, highest ordinary SGPR touched by any fixed Operand/Definition of ACO's typed post-RA instructions + 1) measured inside ACO (aco_interface.cpp export_measure_typed_regs); allocated = 108 hardware total - 2 VCC (ac_gpu_info.c:253, aco_ir.h:438-439). VCC/TTMP/EXEC/M0 are special registers outside these counts; the ACO ra max-used including VCC is kept in sgprs_ra_max_used_incl_vcc.");
    }
    Ok(())
}

fn tree_files(root: &Path, dir: &Path, out: &mut BTreeMap<String, Value>) -> Res<()> {
    let mut entries: Vec<fs::DirEntry> = fs::read_dir(dir)
        .map_err(err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let p = e.path();
        if p.is_dir() {
            tree_files(root, &p, out)?;
        } else if p.is_file() {
            let rel = p.strip_prefix(root).unwrap().to_string_lossy().into_owned();
            if rel == "artifact.json" {
                continue;
            }
            out.insert(
                rel,
                json!({"sha256": sha256_file(&p)?, "bytes": fs::metadata(&p).map_err(err)?.len()}),
            );
        }
    }
    Ok(())
}

/// Compile the SPIR-V through the patched RADV in the host and return the single compute capture dir.
pub fn host_compile(
    host: &Path,
    drv: &DriverEnv,
    spv: &Path,
    k: &Value,
    device_arg: &[String],
    work: &Path,
    extra: &[String],
) -> Res<(PathBuf, BTreeMap<String, String>, Vec<String>)> {
    let capture_tmp = work.join("capture-tmp");
    if capture_tmp.exists() {
        fs::remove_dir_all(&capture_tmp).map_err(err)?;
    }
    fs::create_dir_all(&capture_tmp).map_err(err)?;
    let local: Vec<String> = jarr(k, "local_size")?
        .iter()
        .map(|v| v.to_string())
        .collect();
    let mut argv: Vec<String> = vec![
        "compile".into(),
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
    ];
    argv.extend(device_arg.iter().cloned());
    argv.extend(extra.iter().cloned());
    let mut cmd = Command::new(host);
    cmd.args(&argv);
    let env = drv.apply(&mut cmd, &capture_tmp);
    let out = cmd.output().map_err(|e| format!("spawn host: {e}"))?;
    write_file(&work.join("host.stderr.txt"), &out.stderr)?;
    if !out.status.success() {
        return Err(format!(
            "vk_aco_host failed ({}):\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let mut dirs: Vec<PathBuf> = fs::read_dir(&capture_tmp)
        .map_err(err)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    if dirs.len() != 1 {
        return Err(format!(
            "expected exactly one RADV compute capture in {}, found {} (a pipeline-cache hit or an extra compile invalidates the export)",
            capture_tmp.display(), dirs.len()
        ));
    }
    Ok((dirs.remove(0), env, argv))
}

pub fn spec_kernel<'a>(spec: &'a Value, name: &str) -> Res<&'a Value> {
    spec["kernels"]
        .get(name)
        .ok_or_else(|| format!("kernel `{name}` not in spec"))
}

pub fn layout_hashes(k: &Value) -> Res<(Value, String, Value, String)> {
    let spec_const = json!({"entries": [], "data_hex": ""});
    let layout = json!({
        "descriptor_set_layouts": [],
        "push_constant_ranges": [{"stage_flags": ["COMPUTE"], "offset": 0, "size": ju64(k, "push.size")?}],
        "required_subgroup_size": ju64(k, "subgroup.required")?,
        "require_full_subgroups": jbool(k, "subgroup.require_full")?,
        "local_size": k["local_size"],
    });
    let sh = canon_sha256(&spec_const);
    let lh = canon_sha256(&layout);
    Ok((spec_const, sh, layout, lh))
}

pub fn cmd_export(a: &Args) -> Res<()> {
    let spec_path = a
        .opt("spec")
        .map(PathBuf::from)
        .unwrap_or_else(default_spec);
    let spec = read_json(&spec_path)?;
    let kname = a.req("kernel")?;
    let k = spec_kernel(&spec, kname)?;
    let out = a.path("out")?;
    let mesa_src = a.path("mesa-src")?;
    let mesa_build = a.path("mesa-build")?;
    let host = a.path("host")?;
    let offline = a.flag("offline-shim");
    let device_arg: Vec<String> = if offline {
        vec!["--offline-shim".into()]
    } else {
        vec!["--pci-bdf".into(), a.req("pci-bdf")?.to_owned()]
    };
    if out.exists() && fs::read_dir(&out).map_err(err)?.next().is_some() {
        return Err(format!("{} exists and is not empty", out.display()));
    }
    fs::create_dir_all(&out).map_err(err)?;

    // ------------------------------------------------------------ pin / tree / driver identity
    let pin = mesa::load_pin()?;
    let tree = mesa::verify_tree(&mesa_src, &pin)?;
    let drv = DriverEnv::from_build(&mesa_build, offline)?;
    let driver_sha = sha256_file(&drv.driver_lib)?;

    // ------------------------------------------------------------ GLSL -> SPIR-V
    let src_rel = jstr(k, "source")?;
    let src_path = spec_path.parent().unwrap().join(src_rel);
    let src_sha = sha256_file(&src_path)?;
    let glslang = which("glslangValidator")?;
    let glslang_id = tool_identity(&glslang, "--version", 3)?;
    let spirv_val = which("spirv-val")?;
    let spirv_dis = which("spirv-dis")?;
    let spv = out.join("shader.spv");
    let target_env = jstr(&spec, "glslang.target_env")?;
    let glslang_argv: Vec<String> = vec![
        "-V".into(),
        "--target-env".into(),
        target_env.into(),
        "-S".into(),
        jstr(&spec, "glslang.stage")?.into(),
        "-e".into(),
        jstr(&spec, "glslang.entry")?.into(),
        src_path.file_name().unwrap().to_string_lossy().into_owned(),
        "-o".into(),
        spv.display().to_string(),
    ];
    run(Command::new(&glslang)
        .current_dir(src_path.parent().unwrap())
        .args(&glslang_argv))?;
    run(Command::new(&spirv_val)
        .args(["--target-env", target_env])
        .arg(&spv))?;
    let dis = run(Command::new(&spirv_dis).arg(&spv))?;
    let dis_text = String::from_utf8_lossy(&dis.stdout).into_owned();
    for needle in [
        "OpExecutionMode %main DenormPreserve 32",
        "OpExecutionMode %main SignedZeroInfNanPreserve 32",
        "PhysicalStorageBufferAddresses",
    ] {
        if !dis_text.contains(needle) {
            return Err(format!(
                "SPIR-V lacks required `{needle}` (float controls / BDA did not reach the module)"
            ));
        }
    }
    let spv_sha = sha256_file(&spv)?;

    // ------------------------------------------------------------ SPIR-V -> NIR -> ACO via real RADV
    let host_sha = sha256_file(&host)?;
    let work = out.join("capture");
    fs::create_dir_all(&work).map_err(err)?;
    let (cap_dir, env, host_argv) = host_compile(&host, &drv, &spv, k, &device_arg, &work, &[])?;
    let cap = read_json(&cap_dir.join("capture.json"))?;
    let code = fs::read(cap_dir.join("shader.bin")).map_err(err)?;
    if code.len() as u64 != ju64(&cap, "code_size")? {
        return Err("shader.bin size != capture.code_size".into());
    }
    let mut analysis = abi::analyze(k, &cap)?;
    let exec_size = ju64(&cap, "exec_size")? as usize;
    let llvm_mc = find_llvm_mc(a)?;
    let llvm_mc_id = tool_identity(&llvm_mc, "--version", 3)?;
    let asm = disassemble(&llvm_mc, &analysis.arch, analysis.wave, &code[..exec_size])?;
    apply_body_reads(&mut analysis, &cap)?;

    // move capture files next to the artifact
    for f in ["capture.json", "final.nir.txt", "aco.disasm.txt"] {
        fs::copy(cap_dir.join(f), work.join(f)).map_err(err)?;
    }
    fs::remove_dir_all(work.join("capture-tmp")).map_err(err)?;
    write_file(&out.join("shader.bin"), &code)?;

    // ------------------------------------------------------------ sidecar, rawspec, dump
    let abi_bytes = canon(&analysis.abi);
    write_file(&out.join("shader.abi.json"), &abi_bytes)?;
    let abi_sha = sha256_hex(&abi_bytes);
    let bin_sha = sha256_hex(&code);
    let exec_sha = sha256_hex(&code[..exec_size]);
    let trailing_len = code.len() - exec_size;
    let rawspec = json!({
        "schema": "pm-raw-kernel-spec-v1",
        "target": {"arch": analysis.arch, "xnack": "any", "sramecc": "any", "abi_version": 0},
        "symbol": jstr(k, "symbol")?, "wave": analysis.wave,
        "binary": "shader.bin", "binary_sha256": bin_sha,
        "exec": {"offset": 0, "len": exec_size}, "exec_sha256": exec_sha,
        "sidecar": "shader.abi.json", "sidecar_sha256": abi_sha,
        "resources": analysis.resources, "fp": analysis.fp,
        "relocations": analysis.relocations, "entry": analysis.entry,
    });
    let rawspec_bytes = canon(&rawspec);
    write_file(&out.join("shader.rawspec.json"), &rawspec_bytes)?;

    let stats = fs::read_to_string(work.join("host-stats.txt")).map_err(err)?;
    let local: Vec<String> = jarr(k, "local_size")?
        .iter()
        .map(|v| v.to_string())
        .collect();
    let dump = format!(
        "workgroup_size: {}\nwave_size: {}\n# generated by tools/aco-export: instructions disassembled from shader.bin[0..{exec_size}] by ROCm llvm-mc; statistics from RADV VK_KHR_pipeline_executable_properties\n\ndisasm:\n{asm}\n\n{stats}",
        local.join(", "), analysis.wave
    );
    write_file(&out.join("shader.aco.dump"), dump.as_bytes())?;

    // ------------------------------------------------------------ manifest
    let identity = read_json(&work.join("identity.json"))?;
    let (spec_const, spec_hash, layout, layout_hash) = layout_hashes(k)?;
    let patch = mesa::patch_path(&pin)?;
    let mut files = BTreeMap::new();
    tree_files(&out, &out, &mut files)?;
    let radiowave_argv = vec![
        "oracle".to_string(),
        "aco".into(),
        "--input".into(),
        "shader.aco.dump".into(),
        "--input-artifact".into(),
        "shader.spv".into(),
        "--kernel".into(),
        jstr(k, "symbol")?.into(),
        "--arch".into(),
        analysis.arch.clone(),
        "--wavefront".into(),
        analysis.wave.to_string(),
        "--workgroup".into(),
        local.join(","),
        "--compiler-version".into(),
        format!("RADV/ACO {} {}", jstr(&pin, "tag")?, jstr(&pin, "commit")?),
        // no --output: the report goes to stdout so nothing is ever written inside the artifact directory
    ];
    let artifact = json!({
        "schema": ARTIFACT_SCHEMA,
        "kernel": kname, "symbol": jstr(k, "symbol")?, "entry": jstr(&spec, "glslang.entry")?,
        "files": files,
        "provenance": {
            "compiler_sha256": driver_sha,
            "mesa_patch_sha256": sha256_file(&patch)?,
            "mesa_commit": jstr(&pin, "commit")?,
        },
        "source": {"path": src_rel, "sha256": src_sha, "glslang": glslang_id, "glslang_argv": glslang_argv, "target_env": target_env},
        "spirv": {"file": "shader.spv", "sha256": spv_sha, "validator": tool_identity(&spirv_val, "--version", 2)?,
                  "float_controls": ["DenormPreserve 32", "SignedZeroInfNanPreserve 32"], "addressing": "PhysicalStorageBufferAddresses (BDA, no descriptor sets)"},
        "compiler": {
            "kind": "radv-aco", "pipeline": "SPIR-V -> NIR (radv_shader_spirv_to_nir/radv_postprocess_nir) -> aco_compile_shader (ACO) -> radv_aco_build_shader_binary; capture at radv_shader.c:shader_compile after radv_postprocess_binary_config",
            "llvm_in_compiler": false, "llvm_note": "pin.json meson_options: -Dllvm=disabled; capture.json exists only because the ACO callback ran",
            "mesa": {"tag": pin["tag"], "commit": pin["commit"], "upstream": pin["upstream"], "tree": tree},
            "patch": {"file": format!("tools/aco-export/mesa/{}", jstr(&pin, "patch.file")?), "sha256": sha256_file(&patch)?, "patched_files_sha256": pin["patched_files_sha256"]},
            "meson_options": pin["meson_options"], "build_targets": pin["build_targets"],
            "driver_library": {"path": drv.driver_lib.display().to_string(), "sha256": driver_sha},
            "icd_json": {"path": drv.icd_json.display().to_string(), "sha256": sha256_file(&drv.icd_json)?},
            "host": {"path": host.display().to_string(), "sha256": host_sha, "source_sha256": sha256_file(&tool_dir().join("host/vk_aco_host.c"))?, "argv": host_argv},
            "environment": env,
            "aco_options": cap["aco_options"], "aco_info": cap["aco_info"], "radv_options": cap["radv_options"], "stage_key": cap["stage_key"],
            "exporter": {"name": "aco-export", "version": env!("CARGO_PKG_VERSION")},
        },
        "device": {"identity": identity, "selection": if offline { "offline_shim" } else { "pci_bdf" },
                   "hardware_oracle": !offline, "label": if offline { "OFFLINE drm-shim compile; no hardware identity" } else { "real device compile" },
                   "physical_device": cap["physical_device"]},
        "target": {"arch": analysis.arch, "wave": analysis.wave, "local_size": k["local_size"],
                   "chip": {"family_name": cap["physical_device"]["family_name"], "family": cap["physical_device"]["family"], "gfx_level": cap["physical_device"]["gfx_level"], "name": cap["physical_device"]["name"]}},
        "pipeline": {"specialization": spec_const, "specialization_sha256": spec_hash, "layout": layout, "layout_sha256": layout_hash,
                     "subgroup": k["subgroup"], "pipeline_create_flags": ["CAPTURE_STATISTICS", "CAPTURE_INTERNAL_REPRESENTATIONS"],
                     "stage_flags": ["REQUIRE_FULL_SUBGROUPS"], "entry": "main"},
        "binary": {
            "file": "shader.bin", "sha256": bin_sha, "bytes": code.len(),
            "executable": {"offset": 0, "len": exec_size, "sha256": exec_sha},
            "trailing": {"offset": exec_size, "len": trailing_len, "sha256": if trailing_len > 0 { json!(sha256_hex(&code[exec_size..])) } else { Value::Null }},
            "symbols": cap["symbols"], "relocations": [],
        },
        "config": analysis.config,
        "abi": {"file": "shader.abi.json", "sha256": abi_sha, "schema": "pm-aco-abi-v1"},
        "rawspec": {"file": "shader.rawspec.json", "sha256": sha256_hex(&rawspec_bytes), "schema": "pm-raw-kernel-spec-v1"},
        "bindings": {"push_fields": k["push"]["fields"], "access_extents": k["bindings"]},
        "cache_scope_robustness": {
            "robustness": {"radv_options": cap["radv_options"], "stage_key": cap["stage_key"], "note": "no robustBufferAccess features enabled by the host; BDA accesses are unchecked"},
            "cache_scope": "plain global (BDA) loads/stores as selected by ACO; no coherent/volatile qualifiers in the source; no vulkan memory model",
        },
        "dump": {"file": "shader.aco.dump", "disassembler": llvm_mc_id,
                 "note": "Mesa print_asm is unusable on gfx1201 with the pinned Mesa LLVM 18 (aco-shim README); instruction text is ROCm llvm-mc over the executable range only"},
        "radiowave": {"argv": radiowave_argv, "cwd": "<artifact dir>",
                      "note": "existing CLI (crates/radiowave/src/main.rs:803): radiowave oracle aco --input shader.aco.dump --input-artifact shader.spv ..."},
        "reference_receipt": "separate: `aco-export reference` writes aco-vulkan-ref-v1; not part of static artifact identity",
    });
    write_file(&out.join("artifact.json"), &canon(&artifact))?;
    println!("artifact: {}", out.display());
    println!(
        "artifact.json canonical sha256: {}",
        canon_sha256(&artifact)
    );
    println!("shader.spv sha256: {spv_sha}\nshader.bin sha256: {bin_sha}\nexecutable sha256: {exec_sha} ({exec_size} bytes of {})", code.len());
    println!(
        "radiowave: cd {} && radiowave {}",
        out.display(),
        radiowave_argv.join(" ")
    );
    Ok(())
}

/// `aco-export oracle --artifact DIR --radiowave BIN --out FILE`: run the documented existing radiowave command
/// (stdout = the report) and write the report to an explicit path OUTSIDE the (immutable) artifact directory.
pub fn cmd_oracle(a: &Args) -> Res<()> {
    let dir = a.path("artifact")?;
    let out = a.path("out")?;
    let canon_dir = fs::canonicalize(&dir).map_err(err)?;
    let out_parent = out.parent().ok_or("--out has no parent")?;
    fs::create_dir_all(out_parent).map_err(err)?;
    if fs::canonicalize(out_parent)
        .map_err(err)?
        .starts_with(&canon_dir)
    {
        return Err(
            "--out must not be inside the artifact directory (artifacts are immutable)".into(),
        );
    }
    let art = read_json(&dir.join("artifact.json"))?;
    let argv: Vec<String> = jarr(&art, "radiowave.argv")?
        .iter()
        .map(|v| v.as_str().unwrap_or_default().to_owned())
        .collect();
    let bin = a.path("radiowave")?;
    let o = run(Command::new(&bin).current_dir(&dir).args(&argv))?;
    write_file(&out, &o.stdout)?;
    println!(
        "radiowave oracle report: {} sha256 {}",
        out.display(),
        sha256_file(&out)?
    );
    Ok(())
}
