use hipfire_isa::V;
use hipfire_isa::kernels::gemm_uk::{Chain, Iu4, FragmentLayout, Prefetch};
#[cfg(feature = "toolchain")]
use hipfire_isa::{Arch, Builder, KernelSpec, KernargLayout, RegPlan, reg::Live, kernels::gemm_uk::Iu8};
#[cfg(feature = "toolchain")]
use peacemaker_author::{Gfx1100, Gfx1151, Gfx1201, MmaIu4, Workgroup};

#[test]
fn rejects_overlapping_independent_outputs_and_wrong_prefetch_distance() {
    assert!(Chain::<2, Iu4>::new([V(0), V(4)], V(16)).is_err());
    assert!(Chain::<0, Iu4>::new([], V(16)).is_err());
    assert!(Chain::<2, Iu4>::new([V(0), V(8)], V(0)).is_err());
    assert!(Prefetch::<2>::consume(1, Prefetch::<2>::issue(0, 42)).is_err());
    assert_eq!(Prefetch::<2>::consume(5, Prefetch::<2>::issue(3, 42)).unwrap(), 42);
    let layout = FragmentLayout { rows: 256, k_slices: 8, row_bytes: 8 };
    assert_eq!(layout.offset(255, 7).unwrap(), 16376);
    assert!(layout.offset(256, 0).is_err());
}

#[cfg(feature = "toolchain")]
#[test]
fn iu8_chain_passes_builder_and_llvm_mc() {
    use hipfire_isa::toolchain::Toolchain;
    use std::{fs, process::Command};
    fn assembled<T: MmaIu4, const W: u8>(arch: Arch) where Iu8: hipfire_isa::kernels::gemm_uk::MmaKind<T, Input = V<W>> {
    let mut regs = RegPlan::new(32, 8).unwrap();
    let a = regs.v::<W>("a", 0, Live::Whole).unwrap();
    let x = regs.v::<W>("x", 4, Live::Whole).unwrap();
    let c0 = regs.v::<8>("c0", 8, Live::Whole).unwrap();
    let c1 = regs.v::<8>("c1", 16, Live::Whole).unwrap();
    let seed = regs.v::<8>("seed", 24, Live::Whole).unwrap();
    let mut b = Builder::new(KernelSpec { kernel_id: "iu8_chain".into(), variant: "probe".into(),
        arch, symbol: "iu8_chain".into(), kernargs: KernargLayout::new(8), user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false, workgroup_size: 32, group_segment_fixed_size: 0, wave32: true, cu_mode: false }, regs);
    let mut wg = Workgroup::<T, Builder>::new(&mut b).unwrap();
    let chain = Chain::<2, Iu8>::new([c0, c1], seed).unwrap();
    chain.step(&mut wg, a, [x, x], true).unwrap();
    chain.step(&mut wg, a, [x, x], false).unwrap();
    let end = wg.exit(".Lend").unwrap();
    wg.end(end).unwrap();
    let e = b.finish().unwrap();
    let dir = std::env::temp_dir().join(format!("hipfire-iu8-chain-{}-{}", arch.name(), std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let source = dir.join("chain.s");
    let object = dir.join("chain.o");
    fs::write(&source, e.s_text).unwrap();
    let cpu = format!("-mcpu={}", arch.name());
    let output = Command::new(Toolchain::default().llvm_mc).args(["-triple=amdgcn-amd-amdhsa", &cpu, "-filetype=obj"])
        .arg(&source).arg("-o").arg(&object).output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    fs::remove_dir_all(&dir).unwrap();
    assert!(output.status.success(), "{stderr}");
    }
    assembled::<Gfx1100, 4>(Arch::Gfx1100);
    assembled::<Gfx1151, 4>(Arch::Gfx1151);
    assembled::<Gfx1201, 2>(Arch::Gfx1201);
}
