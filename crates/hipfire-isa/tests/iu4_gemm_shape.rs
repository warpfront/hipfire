//! Offline gates for the builder-emitted iu4 GEMM (plan §4): the hot-loop
//! `IsaShapeContract` of every product point, the 192-VGPR occupancy
//! ceiling, deterministic emission, the loop wait-ledger fix point, the
//! imported SiLU region, and assembly of every emitted module.
use hipfire_isa::Arch;
use hipfire_isa::kernels::iu4_gemm::{self, Cacc, Epi, Fold, Spec, Tile, region::{self, Region}};
use serde_json::Value;
use std::io::Write;
use std::process::{Command, Stdio};

const MC: &str = "/opt/rocm/core-10.0/lib/llvm/bin/llvm-mc";
const POINTS: [(Tile, &str); 2] = [(Tile::T128x128x8, "128x128x8"), (Tile::T256x128x16, "256x128x16")];
const EPIS: [Epi; 3] = [Epi::Set, Epi::Add, Epi::GateUpSilu];

fn contract(tile: &str) -> Value {
    let path = format!("{}/kernels/iu4_gemm.k128-{tile}-cacc1.hotloop.contract.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// The peacemaker count semantics: exact number or {min,max}; a trailing
/// `*` sums every mnemonic with that prefix; absent names count as zero.
fn check_counts(census: &std::collections::BTreeMap<String, u32>, counts: &Value) -> Result<(), String> {
    for (name, bound) in counts.as_object().unwrap() {
        let actual: u32 = match name.strip_suffix('*') {
            Some(prefix) => census.iter().filter(|(m, _)| m.starts_with(prefix)).map(|(_, n)| n).sum(),
            None => census.get(name).copied().unwrap_or(0),
        };
        let ok = match bound {
            Value::Number(n) => u64::from(actual) == n.as_u64().unwrap(),
            Value::Object(o) => o.get("min").is_none_or(|m| u64::from(actual) >= m.as_u64().unwrap())
                && o.get("max").is_none_or(|m| u64::from(actual) <= m.as_u64().unwrap()),
            _ => false,
        };
        if !ok { return Err(format!("{name}: observed {actual}, contract {bound}")) }
    }
    Ok(())
}

fn assemble(text: &str) -> Result<(), String> {
    let mut child = Command::new(MC).args(["-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1201", "-filetype=obj", "-o", "/dev/null"])
        .stdin(Stdio::piped()).stderr(Stdio::piped()).spawn().map_err(|e| e.to_string())?;
    child.stdin.take().unwrap().write_all(text.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    if out.status.success() && out.stderr.is_empty() { Ok(()) } else { Err(String::from_utf8_lossy(&out.stderr).into_owned()) }
}

#[test]
fn hot_loop_meets_shape_contract_for_every_product_symbol() {
    for (tile, name) in POINTS {
        let c = contract(name);
        for epi in EPIS {
            let e = iu4_gemm::emit(Spec { fold: Fold::K128, tile, cacc: Cacc::One, epi, arch: Arch::Gfx1201 }).unwrap();
            let census = iu4_gemm::hot_loop_census(&e.s_text);
            check_counts(&census, &c["counts"]).unwrap_or_else(|m| panic!("{name} {epi:?}: {m}"));
            for f in c["forbidden"].as_array().unwrap() {
                let f = f.as_str().unwrap();
                assert!(!census.keys().any(|m| m == f || f.strip_suffix('*').is_some_and(|p| m.starts_with(p))), "{name} {epi:?}: forbidden {f}");
            }
            assert!(e.shape.next_free_vgpr <= c["vgpr_max"].as_u64().unwrap() as u16, "{name} {epi:?}: {} VGPRs", e.shape.next_free_vgpr);
        }
    }
}

#[test]
fn register_plans_keep_occupancy_ceiling() {
    for (tile, vgprs) in [(Tile::T128x128x8, 187), (Tile::T256x128x16, 186)] {
        for epi in EPIS {
            let e = iu4_gemm::emit(Spec { fold: Fold::K128, tile, cacc: Cacc::One, epi, arch: Arch::Gfx1201 }).unwrap();
            assert_eq!(e.shape.next_free_vgpr, vgprs, "{tile:?} {epi:?}");
            assert!(e.shape.next_free_vgpr <= iu4_gemm::spec::VGPR_CEILING);
        }
    }
}

#[test]
fn emission_is_deterministic_and_loop_ledger_reaches_fixed_point() {
    for (tile, _) in POINTS {
        let (a, text_a, proof_a) = iu4_gemm::emit_module(Fold::K128, tile, Cacc::One, Arch::Gfx1201).unwrap();
        let (_, text_b, proof_b) = iu4_gemm::emit_module(Fold::K128, tile, Cacc::One, Arch::Gfx1201).unwrap();
        assert_eq!(text_a, text_b);
        assert_eq!(proof_a.s_text_sha256, proof_b.s_text_sha256);
        for e in &a {
            assert!(e.proof.loop_fixpoints.iter().any(|l| l.head == ".Liu4_k_loop" && l.iterations == 2));
        }
    }
}

#[test]
fn every_module_assembles_with_zero_diagnostics() {
    for (tile, _) in POINTS {
        let (_, text, _) = iu4_gemm::emit_module(Fold::K128, tile, Cacc::One, Arch::Gfx1201).unwrap();
        assemble(&text).unwrap_or_else(|e| panic!("{tile:?}: {e}"));
    }
}

#[test]
fn closed_and_dropped_axes_are_rejected() {
    for spec in [
        Spec { fold: Fold::K256Shared, ..Spec::control(Epi::Set) },
        Spec { fold: Fold::K256Pow2, ..Spec::control(Epi::Set) },
        Spec { cacc: Cacc::Two, ..Spec::control(Epi::Set) },
        Spec { arch: Arch::Gfx1100, ..Spec::control(Epi::Set) },
    ] {
        assert!(iu4_gemm::emit(spec).is_err(), "{spec:?}");
    }
}

#[test]
fn silu_region_is_the_hipcc_dag() {
    let r = Region::silu().unwrap();
    assert_eq!((r.temps, r.masks), (region::SILU_TEMPS, region::SILU_MASKS));
    let m = r.mnemonics();
    // expf: range reduction, v_exp_f32, ldexp, two saturating selects;
    // then IEEE division with the div_scale/div_fmas/div_fixup sequence.
    for (op, n) in [("v_exp_f32_e32", 1), ("v_ldexp_f32", 1), ("v_cndmask_b32_e64", 2), ("v_div_scale_f32", 2),
        ("v_div_fmas_f32", 1), ("v_div_fixup_f32", 1), ("v_rcp_f32_e32", 1), ("s_wait_alu", 1)] {
        assert_eq!(m.iter().filter(|x| **x == op).count(), n, "{op}");
    }
    assert_eq!(m.last(), Some(&"v_mul_f32_e32"));
    // The region body re-sliced from its own golden is itself.
    let listing = format!("0000 <sym>:\n{}", region::golden_body().lines().map(|l| format!("\t{l} // 0\n")).collect::<String>());
    assert_eq!(region::slice_silu(&listing, "sym").unwrap(), region::golden_body());
}

/// When hipcc is installed, the golden must equal a fresh slice of hipcc's
/// own K1 gate/up object (the region's parse-back from foreign bytes).
#[test]
fn silu_region_matches_fresh_hipcc_object() {
    let hipcc = "/opt/rocm/core-10.0/bin/hipcc";
    if !std::path::Path::new(hipcc).exists() { eprintln!("skip: no {hipcc}"); return }
    let root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("iu4-silu-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("v3.hip");
    let mut text = String::from("#define IU4_SYMMETRIC_FOLD 1\n#define IU4_G12_RASTER 1\n");
    for f in ["kernels/src/block_i4_128_quant.hip", "kernels/src/gemm_mq4g256v2_residual_mmq_iu4_v3.gfx12.hip"] {
        text.push_str(&std::fs::read_to_string(format!("{root}/{f}")).unwrap());
    }
    std::fs::write(&src, text).unwrap();
    let (bundle, co) = (dir.join("v3.hsaco"), dir.join("v3.co"));
    let status = Command::new(hipcc).args(["--genco", "--offload-arch=gfx1201", "-O3", "--no-offload-compress", "-o"])
        .arg(&bundle).arg(&src).status().unwrap();
    assert!(status.success());
    let status = Command::new("/opt/rocm/core-10.0/lib/llvm/bin/clang-offload-bundler")
        .args(["--type=o", "--unbundle", "--targets=hipv4-amdgcn-amd-amdhsa--gfx1201"])
        .arg(format!("--input={}", bundle.display())).arg(format!("--output={}", co.display())).status().unwrap();
    assert!(status.success());
    let dis = Command::new("/opt/rocm/core-10.0/lib/llvm/bin/llvm-objdump").args(["-d", "--mcpu=gfx1201"]).arg(&co).output().unwrap();
    let listing = String::from_utf8(dis.stdout).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(region::slice_silu(&listing, "gemm_mq4g256v2_gate_up_silu_mmq_iu4_v3").unwrap(), region::golden_body());
}
