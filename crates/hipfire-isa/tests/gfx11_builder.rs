//! gfx11 builder target: counter waits, LDS barrier drain, VOPD pairing, the
//! V2C-equivalent SET/ADD/gate-up kernels (gfx1100), the V2B module
//! (gfx1151), and byte identity of the committed builder products.
mod common;
use common::no_llvm;
use hipfire_isa::insn::{Instruction, MemoryClass};
use hipfire_isa::kernels::{fp8_gemm, iu4_gemm, iu4_v2b, iu4_v2c};
use hipfire_isa::lds::Transition;
use hipfire_isa::reg::Live;
use hipfire_isa::vopd::{Operand, VopdF32, VopdOp};
use hipfire_isa::{Arch, Builder, KernargLayout, KernelSpec, RegPlan};
use std::io::Write;
use std::process::{Command, Stdio};

const LLVM: &str = "/opt/rocm/core-10.0/lib/llvm/bin";

fn probe(arch: Arch) -> Builder {
    let mut plan = RegPlan::new(16, 8).unwrap();
    for (name, n) in [
        ("a", 0),
        ("b", 1),
        ("addr", 2),
        ("c", 3),
        ("d", 4),
        ("e", 5),
    ] {
        plan.v::<1>(name, n, Live::Whole).unwrap();
    }
    plan.s::<2>("base", 0, Live::Whole).unwrap();
    Builder::new(
        KernelSpec {
            kernel_id: "probe".into(),
            variant: "default".into(),
            arch,
            symbol: "probe".into(),
            kernargs: KernargLayout::new(8),
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: false,
            workgroup_size: 64,
            group_segment_fixed_size: 0,
            wave32: true,
            cu_mode: false,
        },
        plan,
    )
}
fn v(n: u8) -> hipfire_isa::reg::RegRef {
    hipfire_isa::V::<1>(n).reg()
}

#[test]
fn gfx11_lds_barrier_drains_pending_stores_first() {
    let mut b = probe(Arch::Gfx1100);
    let slot = b.lds_slot("S", 0, 256).unwrap();
    b.ds_store(
        slot,
        Instruction::new("ds_store_b32 v2, v0", vec![], vec![v(2), v(0)])
            .memory(MemoryClass::DsStore),
    )
    .unwrap();
    b.barrier(&[Transition::Ready(slot)]).unwrap();
    let text: Vec<&str> = b
        .program()
        .instructions
        .iter()
        .map(|i| i.text.as_str())
        .collect();
    assert_eq!(
        &text[text.len() - 2..],
        ["s_waitcnt lgkmcnt(0)", "s_barrier"]
    );
    assert!(b.ledger().is_empty());
}

#[test]
fn gfx11_combines_vm_and_lgkm_waits_into_one_s_waitcnt() {
    let mut b = probe(Arch::Gfx1100);
    let slot = b.lds_slot("S", 0, 256).unwrap();
    b.ds_store(
        slot,
        Instruction::new("ds_store_b32 v2, v0", vec![], vec![v(2), v(0)])
            .memory(MemoryClass::DsStore),
    )
    .unwrap();
    b.barrier(&[Transition::Ready(slot)]).unwrap();
    b.push(
        Instruction::new(
            "global_load_b32 v1, v2, s[0:1]",
            vec![v(1)],
            vec![v(2), hipfire_isa::S::<2>(0).reg()],
        )
        .memory(MemoryClass::VmemLoad),
    )
    .unwrap();
    b.push(
        Instruction::new(
            "global_load_b32 v5, v2, s[0:1] offset:4",
            vec![v(5)],
            vec![v(2), hipfire_isa::S::<2>(0).reg()],
        )
        .memory(MemoryClass::VmemLoad),
    )
    .unwrap();
    b.ds_load(
        slot,
        Instruction::new("ds_load_b32 v3, v2", vec![v(3)], vec![v(2)]).memory(MemoryClass::DsLoad),
    )
    .unwrap();
    b.ds_load(
        slot,
        Instruction::new("ds_load_b32 v4, v2 offset:4", vec![v(4)], vec![v(2)])
            .memory(MemoryClass::DsLoad),
    )
    .unwrap();
    // Needs the older VMEM load and the older DS load: vmcnt(1) lgkmcnt(1).
    b.push(Instruction::new(
        "v_add_f32_e32 v0, v1, v3",
        vec![v(0)],
        vec![v(1), v(3)],
    ))
    .unwrap();
    let waits: Vec<&str> = b
        .program()
        .instructions
        .iter()
        .map(|i| i.text.as_str())
        .filter(|t| t.starts_with("s_waitcnt"))
        .collect();
    assert_eq!(waits.last(), Some(&"s_waitcnt vmcnt(1) lgkmcnt(1)"));
    assert_eq!(
        b.waits
            .iter()
            .filter(|w| w.insn == "s_waitcnt vmcnt(1) lgkmcnt(1)")
            .count(),
        2
    );
}

#[test]
fn gfx11_vopd_halves_may_not_read_each_others_destination() {
    let x = VopdOp {
        op: VopdF32::Mul,
        dst: 8,
        src0: Operand::V(1),
        src1: 2,
    };
    let reads_x = VopdOp {
        op: VopdF32::Add,
        dst: 9,
        src0: Operand::Lit(0xcb40_0000),
        src1: 8,
    };
    assert!(hipfire_isa::vopd::validate_pair(Arch::Gfx1100, x, reads_x).is_err());
    assert!(hipfire_isa::vopd::validate_pair(Arch::Gfx1151, x, reads_x).is_err());
    // gfx12 pairing rules are unchanged.
    assert!(hipfire_isa::vopd::validate_pair(Arch::Gfx1201, x, reads_x).is_ok());
    let independent = VopdOp {
        op: VopdF32::Add,
        dst: 9,
        src0: Operand::Lit(0xcb40_0000),
        src1: 9,
    };
    assert!(hipfire_isa::vopd::validate_pair(Arch::Gfx1100, x, independent).is_ok());
}

fn v2c_epi(epi: iu4_v2c::Epi) -> hipfire_isa::Emitted {
    iu4_v2c::emit(iu4_v2c::Spec {
        arch: Arch::Gfx1100,
        epi,
    })
    .unwrap()
}
fn v2c() -> hipfire_isa::Emitted {
    v2c_epi(iu4_v2c::Epi::Set)
}

#[test]
fn v2c_is_deterministic_within_the_occupancy_ceiling() {
    for epi in iu4_v2c::Epi::ALL {
        let (a, b) = (v2c_epi(epi), v2c_epi(epi));
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "{epi:?}");
        assert!(a.shape.next_free_vgpr <= iu4_v2c::VGPR_CEILING, "{epi:?}");
        assert_eq!(a.proof.loop_fixpoints.len(), 1, "{epi:?}");
        assert!(iu4_v2c::emit(iu4_v2c::Spec {
            arch: Arch::Gfx1201,
            epi
        })
        .is_err());
    }
}

#[test]
fn v2c_steady_state_trip_matches_the_v2c_algorithm() {
    // One trip = two K128 epochs; per epoch 64 K16 WMMAs, 2 x (8 ds_swizzle
    // scale broadcasts + 32 mul/add + 16 fmac packets), 8 A rebias XORs,
    // 4 + 4 staging loads, 40 fragment loads, 4 LDS stores and one barrier.
    let census = iu4_v2c::hot_loop_census(&v2c().s_text);
    for (name, n) in [
        ("v_wmma_i32_16x16x16_iu4", 128),
        ("vopd_packets", 192),
        ("v_dual_mul_f32", 128),
        ("v_dual_fmac_f32", 64),
        ("ds_swizzle_b32", 32),
        ("v_mov_b32_dpp", 0),
        ("v_cvt_f32_f16_e64", 2),
        ("v_xor_b32_e32", 16),
        ("global_load_b64", 16),
        ("global_load_b32", 8),
        ("global_load_u16", 2),
        ("ds_load_2addr_b64", 80),
        ("ds_store_2addr_b64", 8),
        ("s_barrier", 2),
        ("valu_slots", 210),
    ] {
        assert_eq!(census.get(name).copied().unwrap_or(0), n, "{name}");
    }
    for forbidden in ["buffer_gl0_inv", "s_nop", "v_nop", "scratch_load_b32"] {
        assert!(!census.contains_key(forbidden), "{forbidden}");
    }
}

/// The K-loop trip between the loop head and its end label.
fn hot_loop(text: &str) -> String {
    let start = text.find(&format!("{}:", iu4_v2c::K_LOOP)).unwrap();
    let end = text.find(&format!("{}:", iu4_v2c::K_LOOP_END)).unwrap();
    text[start..end].to_owned()
}

/// ADD's steady-state K loop is SET's plus, per epoch, one EXEC-gated
/// residual touch (scalar window test and address, one dword load) and
/// nothing else: the fold, staging and fragment streams are unchanged.
#[test]
fn v2c_add_runs_the_set_k_loop_plus_a_gated_touch() {
    let set = iu4_v2c::hot_loop_census(&v2c().s_text);
    let add = iu4_v2c::hot_loop_census(&v2c_epi(iu4_v2c::Epi::Add).s_text);
    for (name, extra) in [
        ("v_wmma_i32_16x16x16_iu4", 0),
        ("vopd_packets", 0),
        ("valu_slots", 0),
        ("ds_load_2addr_b64", 0),
        ("ds_store_2addr_b64", 0),
        ("global_load_b64", 0),
        ("global_load_u16", 0),
        ("global_load_b32", 2),
        ("s_cselect_b32", 2),
        ("s_mov_b32", 2),
        ("s_barrier", 0),
    ] {
        assert_eq!(
            add.get(name).copied().unwrap_or(0),
            set.get(name).copied().unwrap_or(0) + extra,
            "{name}"
        );
    }
    let text = hot_loop(&v2c_epi(iu4_v2c::Epi::Add).s_text);
    assert_eq!(text.matches("s_cselect_b32 exec_lo, -1, 0").count(), 2);
    assert_eq!(text.matches("s_mov_b32 exec_lo, -1").count(), 2);
}

/// Gate/up keeps SET's fully paired fold: per epoch one extra f16 up-scale
/// load and conversion (the pass-1 scales), nothing else.
#[test]
fn v2c_gate_up_trip_is_the_paired_set_fold_plus_up_scales() {
    let set = iu4_v2c::hot_loop_census(&v2c().s_text);
    let silu = iu4_v2c::hot_loop_census(&v2c_epi(iu4_v2c::Epi::Silu).s_text);
    for (name, extra) in [
        ("v_wmma_i32_16x16x16_iu4", 0),
        ("vopd_packets", 0),
        ("ds_swizzle_b32", 0),
        ("v_cvt_f32_f16_e64", 2),
        ("global_load_u16", 2),
        ("global_load_b64", 0),
        ("global_load_b32", 0),
        ("ds_load_2addr_b64", 0),
        ("ds_store_2addr_b64", 0),
        ("s_barrier", 0),
        ("valu_slots", 2),
    ] {
        assert_eq!(
            silu.get(name).copied().unwrap_or(0),
            set.get(name).copied().unwrap_or(0) + extra,
            "{name}"
        );
    }
    // The same 192 fold ops per pass pair mixed on gate/up: per pass 16
    // mul::add + 8 mul::mul (X = mul) and 16 fmac::add + 8 fmac::fmac.
    assert_eq!((silu["v_dual_mul_f32"], silu["v_dual_fmac_f32"]), (96, 96));
}

fn assemble(text: &str, arch: &str) {
    if no_llvm() {
        return;
    }
    let mut child = Command::new(format!("{LLVM}/llvm-mc"))
        .args([
            "-triple=amdgcn-amd-amdhsa",
            &format!("-mcpu={arch}"),
            "-filetype=obj",
            "-o",
            "/dev/null",
        ])
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(text.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success() && out.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn v2c_module_assembles_for_gfx1100_with_zero_diagnostics() {
    assemble(
        &iu4_v2c::module(Arch::Gfx1100, &iu4_v2c::Epi::ALL)
            .unwrap()
            .1,
        "gfx1100",
    )
}

/// The gfx1100 SiLU golden equals a fresh slice of hipcc's own V2C gate/up
/// object (when hipcc is installed): the region's parse-back from foreign
/// bytes, as for the gfx1201 golden.
#[test]
fn gfx1100_silu_region_matches_fresh_hipcc_object() {
    use hipfire_isa::kernels::iu4_gemm::region;
    let hipcc = "/opt/rocm/core-10.0/bin/hipcc";
    if !std::path::Path::new(hipcc).exists() {
        eprintln!("skip: no {hipcc}");
        return;
    }
    let root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
    let dir = std::env::temp_dir().join(format!("v2c-silu-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (bundle, co) = (dir.join("v2c.hsaco"), dir.join("v2c.co"));
    let status = Command::new(hipcc)
        .args([
            "--genco",
            "--offload-arch=gfx1100",
            "-O3",
            "--no-offload-compress",
            "-o",
        ])
        .arg(&bundle)
        .arg(format!(
            "{root}/kernels/src/gemm_mq4g256v2_residual_iu4_v2c.gfx11.hip"
        ))
        .status()
        .unwrap();
    assert!(status.success());
    let status = Command::new(format!("{LLVM}/clang-offload-bundler"))
        .args([
            "--type=o",
            "--unbundle",
            "--targets=hipv4-amdgcn-amd-amdhsa--gfx1100",
        ])
        .arg(format!("--input={}", bundle.display()))
        .arg(format!("--output={}", co.display()))
        .status()
        .unwrap();
    assert!(status.success());
    let dis = Command::new(format!("{LLVM}/llvm-objdump"))
        .args(["-d", "--mcpu=gfx1100"])
        .arg(&co)
        .output()
        .unwrap();
    let listing = String::from_utf8(dis.stdout).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    let slice =
        region::slice_silu_gfx11(&listing, "gemm_mq4g256v2_gate_up_silu_iu4_v2c_gfx11").unwrap();
    assert_eq!(slice, region::body_of(region::SILU_GOLDEN_GFX1100));
    // expf range reduction and the IEEE division, every select mask an SGPR.
    let r = region::Region::silu_gfx1100().unwrap();
    let m = r.mnemonics();
    for (op, n) in [
        ("v_exp_f32_e32", 1),
        ("v_ldexp_f32", 1),
        ("v_cndmask_b32_e64", 2),
        ("v_cmp_nlt_f32_e64", 1),
        ("v_cmp_ngt_f32_e64", 1),
        ("v_div_scale_f32", 2),
        ("v_div_fmas_f32", 1),
        ("v_div_fixup_f32", 1),
        ("v_rcp_f32_e32", 1),
        ("s_mov_b32", 0),
        ("s_wait_alu", 0),
    ] {
        assert_eq!(m.iter().filter(|x| **x == op).count(), n, "{op}");
    }
    assert_eq!((r.masks, m.last().copied()), (2, Some("v_mul_f32_e32")));
}

/// `.text` of the device ELF inside a clang offload bundle (or a bare ELF).
fn text_section(bytes: &[u8]) -> Vec<u8> {
    let at = bytes.windows(4).position(|w| w == b"\x7fELF").unwrap();
    let e = &bytes[at..];
    let u16_ = |o: usize| u16::from_le_bytes(e[o..o + 2].try_into().unwrap()) as usize;
    let u32_ = |o: usize| u32::from_le_bytes(e[o..o + 4].try_into().unwrap()) as usize;
    let u64_ = |o: usize| u64::from_le_bytes(e[o..o + 8].try_into().unwrap()) as usize;
    let (shoff, shentsize, shnum, shstrndx) = (u64_(40), u16_(58), u16_(60), u16_(62));
    let sh = |i: usize| shoff + i * shentsize;
    let names = u64_(sh(shstrndx) + 24);
    (0..shnum)
        .find_map(|i| {
            let name = &e[names + u32_(sh(i))..];
            name.starts_with(b".text\0")
                .then(|| e[u64_(sh(i) + 24)..u64_(sh(i) + 24) + u64_(sh(i) + 32)].to_vec())
        })
        .unwrap()
}

fn link(text: &str, arch: &str, stem: &str) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("hipfire-isa-identity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (s, o, co) = (
        dir.join(format!("{stem}.s")),
        dir.join(format!("{stem}.o")),
        dir.join(format!("{stem}.co")),
    );
    std::fs::write(&s, text).unwrap();
    assert!(Command::new(format!("{LLVM}/llvm-mc"))
        .args([
            "-triple=amdgcn-amd-amdhsa",
            &format!("-mcpu={arch}"),
            "-filetype=obj"
        ])
        .arg(&s)
        .arg("-o")
        .arg(&o)
        .status()
        .unwrap()
        .success());
    assert!(Command::new(format!("{LLVM}/ld.lld"))
        .arg("-shared")
        .arg(&o)
        .arg("-o")
        .arg(&co)
        .status()
        .unwrap()
        .success());
    std::fs::read(co).unwrap()
}

/// The gfx11 target must not move a byte of the embedded gfx1201 products:
/// fresh emission of `_b1`, `_b1s` and the fp8 module links to the exact
/// `.text` of the committed bundles.
#[test]
fn committed_gfx1201_bundles_equal_fresh_emission() {
    if no_llvm() {
        return;
    }
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let token = iu4_gemm::emit_module(
        iu4_gemm::Fold::K128,
        iu4_gemm::Tile::T128x128x8,
        iu4_gemm::Cacc::One,
        iu4_gemm::ALayout::Token,
        Arch::Gfx1201,
    )
    .unwrap()
    .1;
    let slab = iu4_gemm::emit_module(
        iu4_gemm::Fold::K128,
        iu4_gemm::Tile::T128x128x8,
        iu4_gemm::Cacc::One,
        iu4_gemm::ALayout::Slab,
        Arch::Gfx1201,
    )
    .unwrap()
    .1;
    let mut fp8 = Vec::new();
    for act_scale in [fp8_gemm::ActScale::Row, fp8_gemm::ActScale::K128] {
        for epi in [
            fp8_gemm::Epi::Set,
            fp8_gemm::Epi::Add,
            fp8_gemm::Epi::GateUpSilu,
            fp8_gemm::Epi::Qkv,
            fp8_gemm::Epi::Qkvza,
        ] {
            fp8.push(
                fp8_gemm::emit(fp8_gemm::Spec {
                    arch: Arch::Gfx1201,
                    act_scale,
                    epi,
                })
                .unwrap(),
            );
        }
    }
    let fp8 = fp8_gemm::module(&fp8).unwrap().0;
    for (text, bundle) in [
        (token, "gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1"),
        (slab, "gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s"),
        (fp8, "gemm_mq4g256v2_wmma_fp8_gfx12_b1"),
    ] {
        let committed = std::fs::read(format!("{root}/{bundle}.hxaco")).unwrap();
        assert!(
            text_section(&link(&text, "gfx1201", bundle)) == text_section(&committed),
            "{bundle}: fresh gfx1201 emission differs from the committed bundle"
        );
    }
}

fn v2b(epi: iu4_v2b::Epi) -> hipfire_isa::Emitted {
    iu4_v2b::emit(iu4_v2b::Spec {
        arch: Arch::Gfx1151,
        epi,
    })
    .unwrap()
}

#[test]
fn v2b_entries_are_deterministic_and_gfx1151_only() {
    for epi in iu4_v2b::Epi::ALL {
        let (a, b) = (v2b(epi), v2b(epi));
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256);
        assert!(a.shape.next_free_vgpr <= iu4_v2b::VGPR_CEILING);
        for arch in [Arch::Gfx1100, Arch::Gfx1201] {
            assert!(iu4_v2b::emit(iu4_v2b::Spec { arch, epi }).is_err());
        }
    }
}

#[test]
fn v2b_fold_is_fully_vopd_paired_in_every_entry() {
    // One trip = two K128 epochs; per epoch and wave 128 K16 WMMAs, the 384
    // fold ops as 192 packets (128 mul/add, 64 fmac/fmac), 2 converts and 8 A
    // rebias XORs: 202 VALU slots, none of them an unpaired fold op. The 32
    // scale shares are `ds_swizzle_b32` broadcasts, off the VALU port.
    for epi in iu4_v2b::Epi::ALL {
        let census = iu4_v2b::hot_loop_census(&v2b(epi).s_text, epi);
        for (name, n) in [
            ("v_wmma_i32_16x16x16_iu4", 256),
            ("vopd_packets", 384),
            ("v_dual_mul_f32", 256),
            ("v_dual_fmac_f32", 128),
            ("v_mov_b32_dpp", 0),
            ("ds_swizzle_b32", 64),
            ("v_cvt_f32_f16_e64", 4),
            ("v_xor_b32_e32", 16),
            ("valu_slots", 404),
            ("s_barrier", 2),
            ("ds_load_2addr_b64", 160),
            ("ds_store_b64", 16),
            ("global_load_b64", 16),
            ("global_load_b32", 8),
            ("global_load_u16", 4),
        ] {
            assert_eq!(census.get(name).copied().unwrap_or(0), n, "{epi:?} {name}");
        }
        for unpaired in [
            "v_mul_f32_e32",
            "v_add_f32_e32",
            "v_fmac_f32_e32",
            "v_fma_f32",
            "buffer_gl0_inv",
            "s_nop",
            "v_nop",
        ] {
            assert!(!census.contains_key(unpaired), "{epi:?} {unpaired}");
        }
    }
}

#[test]
fn v2b_module_assembles_for_gfx1151_with_zero_diagnostics() {
    assemble(&iu4_v2b::emit_module(Arch::Gfx1151).unwrap().1, "gfx1151")
}

/// The runtime embeds the certified gfx1151 bundle: it must be exactly what
/// the builder emits today.
#[test]
fn committed_gfx1151_v2b_bundle_equals_fresh_emission() {
    if no_llvm() {
        return;
    }
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let committed = std::fs::read(format!("{root}/{}.hxaco", iu4_v2b::MODULE)).unwrap();
    let text = iu4_v2b::emit_module(Arch::Gfx1151).unwrap().1;
    assert!(
        text_section(&link(&text, "gfx1151", iu4_v2b::MODULE)) == text_section(&committed),
        "fresh gfx1151 V2B emission differs from the committed bundle"
    );
}

/// The runtime embeds the certified gfx1100 bundle: it must be exactly what
/// the builder emits today.
#[test]
fn committed_gfx1100_v2c_bundle_equals_fresh_emission() {
    if no_llvm() {
        return;
    }
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../kernels");
    let committed = std::fs::read(format!("{root}/{}.hxaco", iu4_v2c::MODULE)).unwrap();
    let text = iu4_v2c::module(Arch::Gfx1100, &iu4_v2c::Epi::ALL)
        .unwrap()
        .1;
    assert!(
        text_section(&link(&text, "gfx1100", iu4_v2c::MODULE)) == text_section(&committed),
        "fresh gfx1100 V2C emission differs from the committed bundle"
    );
}

#[cfg(feature = "toolchain")]
mod toolchain {
    use super::*;
    use hipfire_isa::{
        ledger_replay, pm_check, profile,
        toolchain::{build, Toolchain},
    };

    /// Every symbol of the product module certifies against its committed
    /// contract: parse-back, exact counts (the linker's inter-kernel
    /// padding is no symbol's `s_nop`), resources, the all-lane LDS bound
    /// and M7's analyses of the linked ELF with no obligation.
    #[test]
    fn v2c_module_passes_gfx11_certification_for_every_symbol() {
        let (emitted, text, _) = iu4_v2c::module(Arch::Gfx1100, &iu4_v2c::Epi::ALL).unwrap();
        for e in &emitted {
            ledger_replay::replay_waits(&e.s_text, Arch::Gfx1100).unwrap();
        }
        let dir = std::env::temp_dir().join(format!("hipfire-isa-v2c-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("v2c.s");
        std::fs::write(&s, &text).unwrap();
        let toolchain = Toolchain::oracle();
        let build = build(&toolchain, &s, &dir.join("v2c.hsaco"), "gfx1100").unwrap();
        for epi in iu4_v2c::Epi::ALL {
            let symbol = iu4_v2c::Spec {
                arch: Arch::Gfx1100,
                epi,
            }
            .symbol();
            assert_eq!(
                pm_check::lds_bounds(&text, &symbol, 8, iu4_v2c::LDS_BYTES).unwrap(),
                iu4_v2c::LDS_BYTES,
                "{epi:?}"
            );
            // A launch allocation smaller than the highest access is rejected.
            assert!(pm_check::lds_bounds(&text, &symbol, 8, iu4_v2c::LDS_BYTES - 8).is_err());
            let m7 = pm_check::m7(&build.elf, "gfx1100", &symbol).unwrap();
            assert_eq!(
                (m7["lift"].as_str(), &m7["obligations"]),
                (Some("byte-exact"), &serde_json::json!({})),
                "{epi:?}"
            );
            let path = format!(
                "{}/kernels/iu4_v2c.gfx1100.{}.contract.json",
                env!("CARGO_MANIFEST_DIR"),
                epi.name()
            );
            let contract = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            hipfire_isa::toolchain::certify(
                &toolchain,
                &build,
                &s,
                "gfx1100",
                &dir.join(format!("{}.manifest.json", epi.name())),
                Some(&contract),
                "test",
                "test",
            )
            .unwrap_or_else(|e| panic!("{epi:?}: {e}"));
        }
    }

    #[test]
    fn v2b_entries_pass_gfx1151_certification_checks() {
        let (_, text, _) = iu4_v2b::emit_module(Arch::Gfx1151).unwrap();
        ledger_replay::replay_waits(&text, Arch::Gfx1151).unwrap();
        let dir = std::env::temp_dir().join(format!("hipfire-isa-v2b-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s = dir.join("v2b.s");
        std::fs::write(&s, &text).unwrap();
        let build = build(&Toolchain::oracle(), &s, &dir.join("v2b.hsaco"), "gfx1151").unwrap();
        for epi in iu4_v2b::Epi::ALL {
            let symbol = iu4_v2b::Spec {
                arch: Arch::Gfx1151,
                epi,
            }
            .symbol();
            assert_eq!(
                pm_check::lds_bounds(&text, &symbol, iu4_v2b::WAVES, iu4_v2b::LDS_BYTES).unwrap(),
                iu4_v2b::LDS_BYTES
            );
            assert!(
                pm_check::lds_bounds(&text, &symbol, iu4_v2b::WAVES, iu4_v2b::LDS_BYTES - 8)
                    .is_err()
            );
            let m7 = pm_check::m7(&build.elf, "gfx1151", &symbol).unwrap();
            assert_eq!(m7["lift"], "byte-exact", "{symbol}");
            assert_eq!(m7["obligations"], serde_json::json!({}), "{symbol}");
        }
    }

    #[test]
    fn gfx11_profile_rewrite_is_verified_and_hazard_neutral() {
        let e = v2c();
        let points = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tools/profile/v2c_set.points.json"
        ))
        .unwrap();
        let cfg: profile::Config = serde_json::from_str(&points).unwrap();
        let (text, map) = profile::instrument(&e.s_text, &cfg, Arch::Gfx1100).unwrap();
        profile::verify(&e.s_text, &text, &map).unwrap();
        assert_eq!(map.cycle_bits, 20);
        assert!(map.vgpr_after <= iu4_v2c::VGPR_CEILING);
        let original = ledger_replay::replay_hazards(
            &profile::kernel_body(&e.s_text, &map.symbol).unwrap(),
            Arch::Gfx1100,
        )
        .unwrap();
        let profiled = ledger_replay::replay_hazards(
            &profile::kernel_body(&text, &map.profiled_symbol).unwrap(),
            Arch::Gfx1100,
        )
        .unwrap();
        assert!(original.is_empty() && profiled.is_empty(), "{profiled:?}");
        assemble(&text, "gfx1100");
        // A store wait the records would count is rejected.
        assert!(profile::instrument(
            &e.s_text
                .replace("\ts_endpgm", "\ts_waitcnt_vscnt null, 0x1\n\ts_endpgm"),
            &cfg,
            Arch::Gfx1100
        )
        .is_err());
    }
}
