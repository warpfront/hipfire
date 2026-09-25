use hipfire_isa::{Arch,Builder,KernelSpec,KernargLayout,RegPlan,Emitted};
use hipfire_isa::{reg::Live,insn::{Instruction,MemoryClass},ledger::Counter,vopd::{VopdOp,VopdF32,Operand}};
use std::{env,fs};

fn probe(arch:Arch)->Result<Emitted,String>{
 let mut regs=RegPlan::new(8,8)?;let mut v=Vec::new();for (name,n) in [("lane",0),("c",1),("w",2),("scale",3),("result",4)] {v.push(regs.v::<1>(name,n,Live::Whole)?)}
 let mut s=Vec::new();for (name,n) in [("arg0",0),("arg1",2),("arg2",4),("arg3",6)] {s.push(regs.s::<2>(name,n,Live::Whole)?)}
 let spec=KernelSpec{kernel_id:"fold_magic".into(),variant:"probe".into(),arch,symbol:"fold_magic".into(),kernargs:KernargLayout::new(32).pointer("out",0).pointer("weight",8).pointer("scale",16).pointer("c",24),user_sgpr_count:2,system_sgpr_workgroup_id_y:false,workgroup_size:1024,group_segment_fixed_size:0,wave32:true};let mut b=Builder::new(spec,regs);
 b.push(Instruction::new("s_load_b256 s[0:7], s[0:1], 0x0",s.iter().map(|r|r.reg()).collect(),vec![s[0].reg()]).memory(MemoryClass::SmemLoad))?;
 b.push(Instruction::new("v_lshlrev_b32_e32 v0, 2, v0",vec![v[0].reg()],vec![v[0].reg()]))?;
 b.wait(Counter::Km,0)?;
 b.clause(|b|{for (dst,ptr) in [(1,2),(2,0),(3,1),(4,3)] {b.push(Instruction::new(format!("global_load_b32 v{dst}, v0, s[{}:{}]",ptr*2,ptr*2+1),vec![v[dst].reg()],vec![v[0].reg(),s[ptr].reg()]).memory(MemoryClass::VmemLoad))?}Ok(())})?;
 b.vopd(VopdOp{op:VopdF32::Add,dst:1,src0:Operand::Lit(0xcb400000),src1:1},VopdOp{op:VopdF32::Mul,dst:2,src0:Operand::V(2),src1:3})?;
 if !arch.gfx12(){return Err("probe fold_magic uses a gfx12-only shared VMEM schedule; gfx11 is table-checked separately".into())}
 b.wait(Counter::Load,0)?;
 b.push(Instruction::new("s_delay_alu instid0(VALU_DEP_1)",vec![],vec![]))?;
 b.push(Instruction::new("v_fmac_f32_e32 v4, v2, v1",vec![v[4].reg()],vec![v[4].reg(),v[2].reg(),v[1].reg()]))?;
 b.push(Instruction::new("global_store_b32 v0, v4, s[6:7]",vec![],vec![v[0].reg(),v[4].reg(),s[3].reg()]).memory(MemoryClass::VmemStore))?;
 b.push(Instruction::new("s_endpgm",vec![],vec![]))?;
 b.finish()
}
fn run()->Result<(),String>{let mut args=env::args().skip(1);if args.next().as_deref()!=Some("emit"){return Err("usage: hipfire-isa emit --kernel fold_magic --arch gfx1201 --out FILE --proof FILE".into())}let mut kernel=None;let mut arch=None;let mut out=None;let mut proof=None;let mut variant=None;while let Some(flag)=args.next(){let value=args.next().ok_or_else(||format!("missing value after {flag}"))?;match flag.as_str(){"--kernel"=>kernel=Some(value),"--arch"=>arch=Some(value.parse::<Arch>()?),"--out"=>out=Some(value),"--proof"=>proof=Some(value),"--variant"=>variant=Some(value),_=>return Err(format!("unknown flag {flag}"))}}
 let kernel=kernel.ok_or("missing --kernel")?;let arch=arch.ok_or("missing --arch")?;let emitted=match kernel.as_str(){"fold_magic"=>probe(arch)?,_=>return Err(format!("kernel {kernel} is not authored; supported probe: fold_magic"))};if let Some(var)=variant {if var!="probe" {return Err("fold_magic supports only variant probe".into())}}
 fs::write(out.ok_or("missing --out")?,emitted.s_text).map_err(|e|e.to_string())?;fs::write(proof.ok_or("missing --proof")?,serde_json::to_vec_pretty(&emitted.proof).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;Ok(())}
fn main(){if let Err(e)=run(){eprintln!("hipfire-isa: {e}");std::process::exit(1)}}
