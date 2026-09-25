//! gfx1201 MQ4V2 E4M3 fragment GEMM, emitted with the checked ISA builder.
//! The 96-byte ABI and all fragment offsets are frozen by fp8-4k5 §11.4–11.5.
pub mod spec;
mod prologue;
mod publish;
mod fold;
mod kloop;
mod epilogue;

pub use spec::{ActScale, Epi, Spec};
use crate::{Arch, Builder, BuilderProof, Emitted, KernelSpec, RegPlan,
    insn::{Instruction, MemoryClass}, reg::{Kind, Live, RegRef}};
use serde::Serialize;
use sha2::{Digest, Sha256};

const BEGIN: &str = ".Lfp8_begin";
const LOOP: &str = ".Lfp8_loop";
const LOOP_END: &str = ".Lfp8_loop_end";
const EPILOGUE: &str = ".Lfp8_epilogue";
const END: &str = ".Lfp8_end";

fn v(n:u8)->RegRef { RegRef { kind:Kind::V,base:n,len:1 } }
fn vr(n:u8,len:u8)->RegRef { RegRef { kind:Kind::V,base:n,len } }
fn s(n:u8)->RegRef { RegRef { kind:Kind::S,base:n,len:1 } }
fn sr(n:u8,len:u8)->RegRef { RegRef { kind:Kind::S,base:n,len } }
fn op(b:&mut Builder,text:impl Into<String>,defs:&[RegRef],uses:&[RegRef])->Result<(),String> {
    b.push(Instruction::new(text,defs.to_vec(),uses.to_vec()))
}
fn mem(b:&mut Builder,text:impl Into<String>,defs:&[RegRef],uses:&[RegRef],class:MemoryClass)->Result<(),String> {
    b.push(Instruction::new(text,defs.to_vec(),uses.to_vec()).memory(class))
}
fn so(b:&mut Builder,text:impl Into<String>,defs:&[u8],uses:&[u8])->Result<(),String> {
    op(b,text,&defs.iter().map(|&n|s(n)).collect::<Vec<_>>(),&uses.iter().map(|&n|s(n)).collect::<Vec<_>>())
}
fn vo(b:&mut Builder,text:impl Into<String>,defs:&[u8],uses:&[u8],su:&[u8])->Result<(),String> {
    op(b,text,&defs.iter().map(|&n|v(n)).collect::<Vec<_>>(),&uses.iter().map(|&n|v(n)).chain(su.iter().map(|&n|s(n))).collect::<Vec<_>>())
}
fn vload(b:&mut Builder,dst:u8,width:u8,voff:u8,srd:u8,soff:Option<u8>,offset:u32)->Result<(),String> {
    let name=match width { 1=>"buffer_load_b32",2=>"buffer_load_b64",4=>"buffer_load_b128",_=>return Err("unsupported VMEM width".into()) };
    let dest=vr(dst,width);
    let sof=soff.map_or("null".to_string(),|n|format!("s{n}"));
    let off=if offset==0 {String::new()} else {format!(" offset:{offset}")};
    let mut uses=vec![v(voff),sr(srd,4)];
    if let Some(n)=soff { uses.push(s(n)); }
    mem(b,format!("{name} {dest}, v{voff}, s[{srd}:{}], {sof} offen{off}",srd+3),&[dest],&uses,MemoryClass::VmemLoad)
}
fn ds_store(b:&mut Builder,slot:usize,data:u8,width:u8,addr:u8,offset:u32)->Result<(),String>{
    let name=match width {1=>"ds_store_b32",2=>"ds_store_b64",_=>return Err("invalid DS store width".into())};
    let off=if offset==0 {String::new()}else{format!(" offset:{offset}")};
    b.ds_store(slot,Instruction::new(format!("{name} v{addr}, {}{off}",vr(data,width)),vec![],vec![v(addr),vr(data,width)]).memory(MemoryClass::DsStore))
}
fn ds_load(b:&mut Builder,slot:usize,data:u8,width:u8,addr:u8,offset:u32)->Result<(),String>{
    let name=match width {1=>"ds_load_b32",2=>"ds_load_b64",_=>return Err("invalid DS load width".into())};
    let off=if offset==0 {String::new()}else{format!(" offset:{offset}")};
    b.ds_load(slot,Instruction::new(format!("{name} {}, v{addr}{off}",vr(data,width)),vec![vr(data,width)],vec![v(addr)]).memory(MemoryClass::DsLoad))
}

// v0:127 accumulator, v128:159 alternating K32 weight units, v160:167
// current activation fragments. v168:175 staging/ratio borrow, v176:191
// addresses and temporary metadata. Prologue and epilogue borrow dead ranges.
fn plan()->Result<RegPlan,String>{
    let mut p=RegPlan::new(spec::VGPR_CEILING,104)?;
    let pro=||Live::Between("entry".into(),BEGIN.into());
    let body=||Live::Between(BEGIN.into(),EPILOGUE.into());
    let epi=||Live::Between(EPILOGUE.into(),END.into());
    for i in 0..16u8 {p.v::<8>("acc",i*8,body())?;p.v::<8>("acc_epilogue",i*8,epi())?;}
    for i in 0..4u8 {p.v::<8>("prologue",i*8,pro())?;}
    for i in 0..8u8 {p.v::<8>("epilogue_temp",128+i*8,epi())?;}
    for i in 0..4u8 {p.v::<8>("weight_ring",128+i*8,body())?;}
    p.v::<8>("activation_fragments",160,body())?;
    p.v::<8>("staging_and_ratio_alias",168,body())?;
    for i in 176..192 {p.v::<1>("hoisted_address",i,body())?;}
    for i in (128..192).step_by(8) {p.v::<8>("prologue_address",i,pro())?;}
    p.s::<2>("kernarg",0,Live::Whole)?;
    for i in (8..32).step_by(8) {p.s::<8>("kernargs",i,Live::Whole)?;}
    for i in (32..68).step_by(4) {p.s::<4>("resource_descriptor",i,Live::Whole)?;}
    for i in 68..72 {p.s::<1>("scalar",i,Live::Whole)?;}
    for i in (72..80).step_by(2) {p.s::<2>("mask_or_scratch",i,Live::Whole)?;}
    for i in 80..104 {p.s::<1>("scalar",i,Live::Whole)?;}
    Ok(p)
}
fn declare_lds(b:&mut Builder)->Result<(),String>{
    for (id,(name,base,len)) in [("A0",0,8192),("A1",8192,8192),("R0",16384,1024),("R1",17408,1024),("D0",18432,512),("D1",18944,512)].into_iter().enumerate(){
        if b.lds.add(name,base,len)?!=id {return Err("LDS slot order".into())}
    }
    Ok(())
}
pub fn emit(spec:Spec)->Result<Emitted,String>{
    spec.validate()?;
    let kernel=KernelSpec{kernel_id:"fp8_gemm".into(),variant:spec.variant(),arch:spec.arch,symbol:spec.symbol(),
        kernargs:Spec::kernargs(),user_sgpr_count:2,system_sgpr_workgroup_id_y:true,
        workgroup_size:256,group_segment_fixed_size:0,wave32:true,cu_mode:false};
    let mut b=Builder::new(kernel,plan()?);
    b.enable_delay_alu();
    declare_lds(&mut b)?;
    prologue::emit(&mut b,spec)?;
    kloop::emit(&mut b,spec)?;
    epilogue::emit(&mut b,spec)?;
    b.label(END)?;
    b.push(crate::insn::Sop::End.encode(Arch::Gfx1201)?)?;
    b.finish()
}
#[derive(Serialize)]
pub struct ModuleProof {pub module:String,pub arch:Arch,pub builder_crate_version:String,pub builder_git_sha:String,
    pub s_text_sha256:String,pub kernels:Vec<BuilderProof>}
pub fn module(emitted:&[Emitted])->Result<(String,ModuleProof),String>{
    let first=emitted.first().ok_or("empty fp8 module")?;
    let mut header=String::new();let mut bodies=String::new();let mut kernels=String::new();let mut tail=String::new();
    for (i,e) in emitted.iter().enumerate(){
        let (code,meta)=e.s_text.split_once(".amdgpu_metadata\n").ok_or("missing metadata")?;
        let (pre,body)=code.split_once(".text\n").ok_or("missing text")?;
        if i==0 {header=pre.into()}else if pre!=header {return Err("target headers differ".into())}
        bodies.push_str(".text\n");
        bodies.push_str(&body.replace(".Lfp8_",&format!(".Lfp8_{}_",e.proof.variant.replace(['-','.'],"_"))));
        let (_,list)=meta.split_once("amdhsa.kernels:\n").ok_or("missing kernels")?;
        let (items,rest)=list.split_once("amdhsa.target:").ok_or("missing target")?;
        kernels.push_str(items);
        if i==0 {tail=format!("amdhsa.target:{rest}")}
    }
    let text=format!("{header}{bodies}.amdgpu_metadata\n---\namdhsa.kernels:\n{kernels}{tail}");
    let proof=ModuleProof{module:Spec::module().into(),arch:first.proof.arch,
        builder_crate_version:first.proof.builder_crate_version.clone(),builder_git_sha:first.proof.builder_git_sha.clone(),
        s_text_sha256:format!("{:x}",Sha256::digest(text.as_bytes())),kernels:emitted.iter().map(|e|e.proof.clone()).collect()};
    Ok((text,proof))
}
pub fn emit_module(arch:Arch,scale:ActScale,epis:&[Epi])->Result<(Vec<Emitted>,String,ModuleProof),String>{
    let emitted=epis.iter().map(|&epi|emit(Spec{arch,act_scale:scale,epi})).collect::<Result<Vec<_>,_>>()?;
    let (text,proof)=module(&emitted)?;Ok((emitted,text,proof))
}
