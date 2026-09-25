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
const USAGE:&str="usage: hipfire-isa emit --kernel fold_magic --arch gfx1201 --out FILE --proof FILE\n       hipfire-isa emit --kernel iu4_gemm --fold k128 --tile 128x128x8|256x128x16 --cacc 1 --epi set|add|silu|all --arch gfx1201 --out FILE --proof FILE\n       hipfire-isa emit --kernel fp8_gemm --scale row|k128|both --epi set|add|silu|qkv|qkvza|all --arch gfx1201 --out FILE --proof FILE\n       hipfire-isa region-import --disassembly OBJDUMP.txt [--symbol gemm_mq4g256v2_gate_up_silu_mmq_iu4_v3]";
/// `--epi all` emits the three epilogue symbols as one module (the product
/// code object the oracle loads); a single epilogue emits one symbol.
fn iu4_gemm(fold:&str,tile:&str,cacc:&str,epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::iu4_gemm::{self,Spec};
 let (fold,tile,cacc)=(fold.parse()?,tile.parse()?,cacc.parse()?);
 if epi=="all"{let (_,text,proof)=iu4_gemm::emit_module(fold,tile,cacc,arch)?;return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))}
 let emitted=iu4_gemm::emit(Spec{fold,tile,cacc,epi:epi.parse()?,arch})?;
 Ok((emitted.s_text,serde_json::to_vec_pretty(&emitted.proof).map_err(|e|e.to_string())?))
}
fn fp8_gemm(scale:&str,epi:&str,arch:Arch)->Result<(String,Vec<u8>),String>{
 use hipfire_isa::kernels::fp8_gemm::{self,Spec,ActScale,Epi};
 if scale=="both" {
     if epi!="all" {return Err("--scale both requires --epi all".into())}
     let epis=[Epi::Set,Epi::Add,Epi::GateUpSilu,Epi::Qkv,Epi::Qkvza];
     let mut emitted=Vec::with_capacity(10);
     for act_scale in [ActScale::Row,ActScale::K128] {
         for &epi in &epis {emitted.push(fp8_gemm::emit(Spec{arch,act_scale,epi})?);}
     }
     let (text,proof)=fp8_gemm::module(&emitted)?;
     return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))
 }
 let scale:ActScale=scale.parse()?;
 if matches!(epi,"all"|"initial") {
     let epis: &[Epi]=if epi=="initial" {&[Epi::Set,Epi::Add]} else {&[Epi::Set,Epi::Add,Epi::GateUpSilu,Epi::Qkv,Epi::Qkvza]};
     let (_,text,proof)=fp8_gemm::emit_module(arch,scale,epis)?;
     return Ok((text,serde_json::to_vec_pretty(&proof).map_err(|e|e.to_string())?))
 }
 let emitted=fp8_gemm::emit(Spec{arch,act_scale:scale,epi:epi.parse()?})?;
 Ok((emitted.s_text,serde_json::to_vec_pretty(&emitted.proof).map_err(|e|e.to_string())?))
}
/// Re-slice the SiLU region from a hipcc disassembly and require it to equal
/// the committed golden the gate/up epilogue instantiates.
fn region_import(mut args:impl Iterator<Item=String>)->Result<(),String>{
 use hipfire_isa::kernels::iu4_gemm::region;
 let mut dis=None;let mut symbol="gemm_mq4g256v2_gate_up_silu_mmq_iu4_v3".to_string();
 while let Some(flag)=args.next(){let value=args.next().ok_or_else(||format!("missing value after {flag}"))?;match flag.as_str(){"--disassembly"=>dis=Some(value),"--symbol"=>symbol=value,_=>return Err(format!("unknown flag {flag}\n{USAGE}"))}}
 let text=fs::read_to_string(dis.ok_or("missing --disassembly")?).map_err(|e|e.to_string())?;
 let slice=region::slice_silu(&text,&symbol)?;
 print!("{slice}");
 if slice!=region::golden_body(){return Err("sliced region differs from kernels/iu4_gemm.silu.region.s".into())}
 region::Region::parse(&slice)?;
 eprintln!("region-import: {symbol} SiLU slice matches the committed golden");
 Ok(())
}
fn run()->Result<(),String>{let mut args=env::args().skip(1);let command=args.next();if command.as_deref()==Some("region-import"){return region_import(args)}if command.as_deref()!=Some("emit"){return Err(USAGE.into())}let mut kernel=None;let mut arch=None;let mut out=None;let mut proof=None;let mut variant=None;let (mut fold,mut tile,mut cacc,mut epi,mut scale)=(None,None,None,None,None);while let Some(flag)=args.next(){let value=args.next().ok_or_else(||format!("missing value after {flag}"))?;match flag.as_str(){"--kernel"=>kernel=Some(value),"--arch"=>arch=Some(value.parse::<Arch>()?),"--out"=>out=Some(value),"--proof"=>proof=Some(value),"--variant"=>variant=Some(value),"--fold"=>fold=Some(value),"--tile"=>tile=Some(value),"--cacc"=>cacc=Some(value),"--epi"=>epi=Some(value),"--scale"=>scale=Some(value),_=>return Err(format!("unknown flag {flag}"))}}
 let kernel=kernel.ok_or("missing --kernel")?;let arch=arch.ok_or("missing --arch")?;
 let (text,proof_json)=match kernel.as_str(){
  "fold_magic"=>{if let Some(var)=variant {if var!="probe" {return Err("fold_magic supports only variant probe".into())}}let emitted=probe(arch)?;(emitted.s_text,serde_json::to_vec_pretty(&emitted.proof).map_err(|e|e.to_string())?)}
  "iu4_gemm"=>iu4_gemm(fold.as_deref().unwrap_or("k128"),tile.as_deref().ok_or("missing --tile")?,cacc.as_deref().unwrap_or("1"),epi.as_deref().ok_or("missing --epi")?,arch)?,
  "fp8_gemm"=>fp8_gemm(scale.as_deref().ok_or("missing --scale")?,epi.as_deref().ok_or("missing --epi")?,arch)?,
  _=>return Err(format!("kernel {kernel} is not authored\n{USAGE}"))};
 fs::write(out.ok_or("missing --out")?,text).map_err(|e|e.to_string())?;fs::write(proof.ok_or("missing --proof")?,proof_json).map_err(|e|e.to_string())?;Ok(())}
fn main(){if let Err(e)=run(){eprintln!("hipfire-isa: {e}");std::process::exit(1)}}
