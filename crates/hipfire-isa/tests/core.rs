use hipfire_isa::{Arch,RegPlan,reg::{Live,V,Vb,Vp,RegRef,Kind},vopd::{self,VopdOp,VopdF32,Operand},insn::{Instruction,Vbuffer,MemoryClass,Sop,Ds},hazard::{Gfx12Sgpr,Pipeline},ledger::{Ledger,Counter}};
use std::{io::Write,process::{Command,Stdio}};
#[test]fn vopd_banks_and_architectural_shared_src1(){let x=VopdOp{op:VopdF32::Mul,dst:0,src0:Operand::V(8),src1:7};let y=VopdOp{op:VopdF32::Mul,dst:1,src0:Operand::V(9),src1:7};assert!(vopd::packet(Arch::Gfx1201,x,y).is_ok());assert!(vopd::packet(Arch::Gfx1100,x,y).is_err());assert!(vopd::packet(Arch::Gfx1151,x,y).is_err());assert!(vopd::packet(Arch::Gfx1201,x,VopdOp{dst:2,..y}).is_err());assert!(vopd::packet(Arch::Gfx1201,x,VopdOp{src0:Operand::V(12),src1:6,..y}).is_err());assert!(vopd::packet(Arch::Gfx1201,x,VopdOp{src1:11,..y}).is_err());assert!(Vb::<0>::checked(1).is_err());assert!(Vp::<0>::checked(1).is_err())}
#[test]fn contiguous_pairs_are_banked_and_opposite(){for base in 0..=248 {for j in [0,2,4,6]{let (x,y)=V::<8>(base).pair(j).unwrap();assert_ne!(x.0%4,y.0%4);assert_ne!(x.0%2,y.0%2)}}}
#[test]fn limits_reject_silent_assembler_wraps(){let r=|kind,base,len|RegRef{kind,base,len};let d=r(Kind::V,0,2);let vo=r(Kind::V,2,1);let srd=r(Kind::S,4,4);assert!(Vbuffer::LoadB64.emit(Arch::Gfx1100,d,vo,srd,None,4095).is_ok());assert!(Vbuffer::LoadB64.emit(Arch::Gfx1100,d,vo,srd,None,4096).is_err());assert!(Vbuffer::LoadB64.emit(Arch::Gfx1201,d,vo,srd,None,(1<<23)-1).is_ok());assert!(Vbuffer::LoadB64.emit(Arch::Gfx1201,d,vo,srd,None,1<<23).is_err());assert!(Sop::Clause(33).encode(Arch::Gfx1201).is_err());assert!(Sop::WaitLoad(64).encode(Arch::Gfx1201).is_err());assert!(Sop::WaitKm(32).encode(Arch::Gfx1201).is_err());assert!(Ds::Load2B64.offset(256,0).is_err());assert!(Arch::Gfx1201.check_mnemonic("s_waitcnt").is_err());assert!(Arch::Gfx1100.check_mnemonic("v_swmmac_i32_16x16x64_iu4").is_err())}
#[test]fn sgpr_hazard_tracks_only_valu_read_pairs(){let mut h=Gfx12Sgpr::default();let s=|base|RegRef{kind:Kind::S,base,len:1};assert!(h.step(Pipeline::Salu,&[],&[s(10)],false,false).is_empty());assert!(h.step(Pipeline::Vmem,&[s(10)],&[],false,false).is_empty());assert!(h.step(Pipeline::Valu,&[s(10)],&[],false,false).is_empty());assert!(h.step(Pipeline::Salu,&[],&[s(10)],false,false).is_empty());assert_eq!(h.step(Pipeline::Salu,&[s(10)],&[],false,false),["depctr_sa_sdst(0)"]);assert!(h.step(Pipeline::Valu,&[],&[s(10)],false,false).is_empty());assert_eq!(h.step(Pipeline::Valu,&[s(10)],&[],false,false),["depctr_va_sdst(0)"]);h.step(Pipeline::Salu,&[],&[s(10)],false,false);h.step(Pipeline::Smem,&[],&[],false,false);assert!(h.step(Pipeline::Valu,&[s(10)],&[],false,false).is_empty())}
#[test]fn ledger_retains_partial_load_and_ds_waits(){let r=|base|RegRef{kind:Kind::V,base,len:1};let mut ledger=Ledger::default();for n in [7,79,91,83]{ledger.record(Arch::Gfx1201,&Instruction::new("global_load_b32",vec![r(n)],vec![]).memory(MemoryClass::VmemLoad))}for (n,expected) in [(7,3),(79,2),(91,1),(83,0)]{let req=ledger.required(&Instruction::new("v_xor",vec![r(100)],vec![r(n)]));assert_eq!(req[0].0,Counter::Load);assert_eq!(req[0].1,expected);ledger.wait(Counter::Load,expected)}for n in [71,75,79]{ledger.record(Arch::Gfx1201,&Instruction::new("ds_load_b64",vec![r(n)],vec![]).memory(MemoryClass::DsLoad))}let req=ledger.required(&Instruction::new("v_wmma",vec![r(3)],vec![r(75)]));assert_eq!(req[0].0,Counter::Ds);assert_eq!(req[0].1,1);ledger.wait(Counter::Ds,1);let req=ledger.required(&Instruction::new("v_wmma",vec![r(3)],vec![r(79)]));assert_eq!(req[0].1,0)}
#[test]fn reg_lifetimes_and_budget(){let mut p=RegPlan::new(8,8).unwrap();p.v::<4>("a",0,Live::Whole).unwrap();assert!(p.v::<2>("alias",2,Live::Whole).is_err());assert!(p.v::<8>("too_large",2,Live::Whole).is_err());assert!(p.s::<4>("srd",2,Live::Whole).is_err())}
#[test]fn assembler_golden_tables(){let mc="/opt/rocm/core-10.0/lib/llvm/bin/llvm-mc";for (arch,file,table) in [("gfx1201",include_str!("golden/gfx1201.txt"),hipfire_isa::insn::GFX1201_GOLDEN),("gfx1100",include_str!("golden/gfx1100.txt"),hipfire_isa::insn::GFX1100_GOLDEN)]{let lines:Vec<_>=file.lines().collect();assert_eq!(lines,table);let mut child=Command::new(mc).args(["-triple=amdgcn-amd-amdhsa",&format!("-mcpu={arch}"),"-show-encoding"]).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();child.stdin.take().unwrap().write_all(file.as_bytes()).unwrap();let output=child.wait_with_output().unwrap();assert!(output.status.success(),"{arch}: {}",String::from_utf8_lossy(&output.stderr));let text=String::from_utf8(output.stdout).unwrap();assert_eq!(text.matches("encoding: [").count(),lines.len(),"{arch} output:\n{text}")}}

#[test]
fn lds_slot_publication_and_split_barrier() {
    use hipfire_isa::lds::{Lds, SlotState, Transition};
    let mut lds = Lds::default();
    let a = lds.add("A0", 0, 1024).unwrap();
    assert!(lds.load(a).is_err());
    lds.store(a).unwrap();
    assert!(lds.barrier_signal(&[Transition::Ready(a)], false).is_err());
    lds.barrier_signal(&[Transition::Ready(a)], true).unwrap();
    assert!(lds.load(a).is_err());
    lds.barrier_wait().unwrap();
    assert_eq!(lds.slots[a].state, SlotState::Published);
    lds.load(a).unwrap();
    lds.barrier(&[Transition::Retire(a)], true).unwrap();
    assert_eq!(lds.slots[a].state, SlotState::Free);
    lds.barrier(&[], true).unwrap();
}

#[test]
fn raw_instruction_immediates_cannot_evade_architecture_table() {
    for (arch, text) in [
        (Arch::Gfx1201, "s_clause 0x40"),
        (Arch::Gfx1201, "s_wait_loadcnt 0x40"),
        (Arch::Gfx1201, "s_wait_kmcnt 0x20"),
        (Arch::Gfx1100, "buffer_load_b64 v[0:1], v2, s[4:7], s8 offen offset:4096"),
        (Arch::Gfx1201, "ds_load_2addr_b64 v[0:3], v1 offset0:256"),
        (Arch::Gfx1100, "v_wmma_i32_16x16x32_iu4 v[0:7], v[8:9], v[10:11], 0"),
    ] {
        assert!(Instruction::new(text, vec![], vec![]).validate(arch).is_err(), "{text}");
    }
}

#[test]
fn gfx1151_uses_the_gfx11_encoding_table() {
    let file = include_str!("golden/gfx1100.txt");
    let mut child = Command::new("/opt/rocm/core-10.0/lib/llvm/bin/llvm-mc")
        .args(["-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1151", "-show-encoding"])
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().unwrap();
    child.stdin.take().unwrap().write_all(file.as_bytes()).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let encoding = String::from_utf8(output.stdout).unwrap();
    assert_eq!(encoding.matches("encoding: [").count(), file.lines().count());
}

#[test]
fn mixed_memory_families_and_gfx11_lgkm_force_zero_wait() {
    let v = |base| RegRef { kind: Kind::V, base, len: 1 };
    let s = |base| RegRef { kind: Kind::S, base, len: 1 };
    let mut gfx12 = Ledger::default();
    gfx12.record(Arch::Gfx1201, &Instruction::new("buffer_load_b32", vec![v(1)], vec![]).memory(MemoryClass::VmemLoad));
    gfx12.record(Arch::Gfx1201, &Instruction::new("global_load_b32", vec![v(2)], vec![]).memory(MemoryClass::VmemLoad));
    assert_eq!(gfx12.required(&Instruction::new("v_add", vec![v(3)], vec![v(1)]))[0].1, 0);

    let mut gfx11 = Ledger::default();
    gfx11.record(Arch::Gfx1100, &Instruction::new("s_load_b32", vec![s(0)], vec![]).memory(MemoryClass::SmemLoad));
    gfx11.record(Arch::Gfx1100, &Instruction::new("ds_load_b32", vec![v(1)], vec![]).memory(MemoryClass::DsLoad));
    gfx11.record(Arch::Gfx1100, &Instruction::new("ds_load_b32", vec![v(2)], vec![]).memory(MemoryClass::DsLoad));
    assert_eq!(gfx11.required(&Instruction::new("v_add", vec![v(3)], vec![v(1)]))[0].1, 0);
}

#[test]
fn builder_combines_simultaneous_load_and_ds_waits() {
    use hipfire_isa::{Builder, KernelSpec, KernargLayout};
    let mut plan = RegPlan::new(8, 8).unwrap();
    let value = plan.v::<1>("value", 0, Live::Whole).unwrap();
    let loaded = plan.v::<1>("loaded", 1, Live::Whole).unwrap();
    let address = plan.v::<1>("address", 2, Live::Whole).unwrap();
    let srd = plan.s::<2>("srd", 0, Live::Whole).unwrap();
    let spec = KernelSpec {
        kernel_id: "wait_probe".into(), variant: "default".into(), arch: Arch::Gfx1201,
        symbol: "wait_probe".into(), kernargs: KernargLayout::new(8),
        user_sgpr_count: 2, workgroup_size: 32, group_segment_fixed_size: 0, wave32: true,
    };
    let mut builder = Builder::new(spec, plan);
    let slot = builder.lds.add("A0", 0, 256).unwrap();
    builder.ds_store(slot, Instruction::new("ds_store_b32 v2, v0", vec![], vec![address.reg(), value.reg()]).memory(MemoryClass::DsStore)).unwrap();
    builder.push(Instruction::new("global_load_b32 v1, v2, s[0:1]", vec![loaded.reg()], vec![address.reg(), srd.reg()]).memory(MemoryClass::VmemLoad)).unwrap();
    builder.push(Instruction::new("v_mov_b32 v0, v1", vec![value.reg()], vec![loaded.reg()])).unwrap();
    let combined: Vec<_> = builder.program.instructions.iter().filter(|i| i.mnemonic()=="s_wait_loadcnt_dscnt").collect();
    assert_eq!(combined.len(), 1);
    assert_eq!(combined[0].text, "s_wait_loadcnt_dscnt 0x0");
    assert_eq!(builder.waits.iter().filter(|p| p.insn==combined[0].text).count(), 2);
}

#[test]
fn wmma_to_wmma_ab_dependency_inserts_nop() {
    use hipfire_isa::{Builder, KernelSpec, KernargLayout};
    let mut plan = RegPlan::new(32, 8).unwrap();
    let first = plan.v::<8>("first", 0, Live::Whole).unwrap();
    let a = plan.v::<2>("a", 8, Live::Whole).unwrap();
    let b = plan.v::<2>("b", 10, Live::Whole).unwrap();
    let second = plan.v::<8>("second", 12, Live::Whole).unwrap();
    let spec = KernelSpec {
        kernel_id: "wmma_probe".into(), variant: "default".into(), arch: Arch::Gfx1201,
        symbol: "wmma_probe".into(), kernargs: KernargLayout::new(8),
        user_sgpr_count: 2, workgroup_size: 32, group_segment_fixed_size: 0, wave32: true,
    };
    let mut builder = Builder::new(spec, plan);
    builder.push(Instruction::new(
        "v_wmma_i32_16x16x32_iu4 v[0:7], v[8:9], v[10:11], 0 neg_lo:[1,1,0]",
        vec![first.reg()], vec![a.reg(), b.reg()],
    )).unwrap();
    builder.push(Instruction::new(
        "v_wmma_i32_16x16x32_iu4 v[12:19], v[0:1], v[10:11], 0 neg_lo:[1,1,0]",
        vec![second.reg()], vec![RegRef { kind: Kind::V, base: 0, len: 2 }, b.reg()],
    )).unwrap();
    assert_eq!(builder.program.instructions[1].text, "v_nop");
    assert_eq!(builder.hazards[0].rule, "WMMA destination feeds next WMMA A/B or SWMMAC index");
}

#[test]
fn workgroup_id_abi_tracks_user_sgpr_count() {
    use hipfire_isa::arch::WgIdSource;
    let gfx12 = Arch::Gfx1201.workgroup_ids(14).unwrap();
    assert_eq!((gfx12.x, gfx12.y), (WgIdSource::Ttmp9, WgIdSource::Ttmp7));
    let gfx11 = Arch::Gfx1100.workgroup_ids(14).unwrap();
    assert_eq!((gfx11.x, gfx11.y), (WgIdSource::Sgpr(14), WgIdSource::Sgpr(15)));
    assert!(Arch::Gfx1151.workgroup_ids(103).is_err());
}

#[test]
fn global_visibility_is_explicit_not_implied_by_lds_barrier() {
    use hipfire_isa::{Builder, KernelSpec, KernargLayout, MemoryScope};
    let spec = KernelSpec {
        kernel_id: "barrier_probe".into(), variant: "default".into(), arch: Arch::Gfx1201,
        symbol: "barrier_probe".into(), kernargs: KernargLayout::new(8),
        user_sgpr_count: 2, workgroup_size: 32, group_segment_fixed_size: 0, wave32: true,
    };
    let mut builder = Builder::new(spec, RegPlan::new(8, 8).unwrap());
    builder.barrier(&[]).unwrap();
    assert_eq!(builder.program.instructions.iter().filter(|i| i.mnemonic()=="global_inv").count(), 0);
    builder.barrier_with_scope(&[], MemoryScope::Workgroup).unwrap();
    assert_eq!(builder.program.instructions.iter().filter(|i| i.mnemonic()=="global_inv").count(), 1);
    assert!(builder.push(Instruction::new("s_barrier_signal -1", vec![], vec![])).is_err());
    builder.push(Instruction::new("s_endpgm", vec![], vec![])).unwrap();
    assert_eq!(builder.finish().unwrap().proof.barriers.len(), 2);
}
