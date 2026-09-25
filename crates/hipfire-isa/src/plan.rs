use crate::{arch::Arch,reg::Range,ledger::WaitProof,hazard::HazardProof,lds::LdsSlot};
use serde::Serialize;

#[derive(Clone,Debug,Serialize)] pub struct Kernarg {pub name:String,pub offset:u32,pub size:u32,pub value_kind:String,pub address_space:Option<String>}
#[derive(Clone,Debug,Serialize)] pub struct KernargLayout {pub size:u32,pub args:Vec<Kernarg>}
impl KernargLayout {
 pub fn new(size:u32)->Self {Self{size,args:vec![]}}
 pub fn pointer(mut self,name:&str,offset:u32)->Self{self.args.push(Kernarg{name:name.into(),offset,size:8,value_kind:"global_buffer".into(),address_space:Some("global".into())});self}
 pub fn hidden(mut self,name:&str,offset:u32,size:u32,value_kind:&str)->Self{self.args.push(Kernarg{name:name.into(),offset,size,value_kind:value_kind.into(),address_space:None});self}
 pub fn validate(&self)->Result<(),String>{let mut end=0;for a in &self.args {if a.offset<end||a.offset.checked_add(a.size).is_none_or(|n|n>self.size){return Err(format!("overlapping/out-of-bounds kernarg {}",a.name))}end=a.offset+a.size}Ok(())}
}
#[derive(Clone,Debug)] pub struct KernelSpec {pub kernel_id:String,pub variant:String,pub arch:Arch,pub symbol:String,pub kernargs:KernargLayout,pub user_sgpr_count:u8,pub workgroup_size:u16,pub group_segment_fixed_size:u32,pub wave32:bool}
#[derive(Clone,Copy,Debug,PartialEq,Eq)] pub enum MemoryScope { LdsOnly, Workgroup }
#[derive(Clone,Debug,Default,Serialize)] pub struct IsaShape {pub instructions:usize,pub valu_slots:usize,pub vopd_pairs:usize,pub wmma:usize,pub ds:usize,pub vmem:usize,pub waits:usize,pub barriers:usize,pub next_free_vgpr:u16,pub next_free_sgpr:u16}
#[derive(Clone,Debug,Serialize)] pub struct ClauseProof {pub start:usize,pub len:usize,pub class:String}
#[derive(Clone,Debug,Serialize)] pub struct BarrierProof {pub pc_index:usize,pub transitions:Vec<String>}
#[derive(Clone,Debug,Serialize)] pub struct LoopFixpoint {pub head:String,pub iterations:u8}
#[derive(Clone,Debug,Serialize)] pub struct BuilderProof {
 pub kernel_id:String,pub variant:String,pub arch:Arch,pub builder_crate_version:String,pub builder_git_sha:String,
 pub reg_plan:Vec<Range>,pub next_free_vgpr:u16,pub next_free_sgpr:u16,pub waits:Vec<WaitProof>,
 pub hazards:Vec<HazardProof>,pub clauses:Vec<ClauseProof>,pub vopd_pairs:usize,pub lds_slots:Vec<LdsSlot>,
 pub barriers:Vec<BarrierProof>,pub loop_fixpoints:Vec<LoopFixpoint>,pub forbidden_mnemonics_checked:Vec<String>,pub s_text_sha256:String,
}
#[derive(Clone,Debug)] pub struct Emitted {pub s_text:String,pub proof:BuilderProof,pub shape:IsaShape}
