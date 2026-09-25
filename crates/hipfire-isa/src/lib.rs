#![forbid(unsafe_code)]
//! An imported `Region<In, Out>` must take ownership of typed live-in `V<N>`/`S<N>`
//! ranges and LDS slot tokens, then return its live-out handles. The seam carries
//! `Ledger` plus EXEC/VCC/SCC/M0 capabilities; it reconciles pending counters
//! and slot states before returning ownership. Foreign bytes become a region
//! only after disassembly parse-back and independent wait-ledger replay.
pub mod arch; pub mod reg; pub mod plan; pub mod ledger; pub mod hazard; pub mod vopd; pub mod lds; pub mod insn; pub mod emit; pub mod aco;
pub mod kernels { pub mod iu4_k1; pub mod iu4_gemm; pub mod fp8_gemm; }
#[cfg(feature="toolchain")] pub mod toolchain;
#[cfg(feature="toolchain")] pub mod ledger_replay;
#[cfg(feature="toolchain")] pub mod audit;
pub use arch::Arch;
pub use reg::{RegPlan,V,S};
pub use plan::{KernelSpec,KernargLayout,Emitted,BuilderProof,IsaShape,MemoryScope};
use insn::{Instruction,Program,MemoryClass};
use ledger::{Ledger,WaitProof,Counter,Reason};
use hazard::{Gfx12Sgpr,Gfx11Hazards,Pipeline,HazardProof};
use lds::{Lds,Transition};
use sha2::{Sha256,Digest};

#[derive(Clone)] pub struct Builder { pub spec:KernelSpec,pub regs:RegPlan,pub program:Program,pub ledger:Ledger,pub lds:Lds,
 pub waits:Vec<WaitProof>,pub hazards:Vec<HazardProof>,pub clauses:Vec<plan::ClauseProof>,pub barriers:Vec<plan::BarrierProof>,pub loop_fixpoints:Vec<plan::LoopFixpoint>,
 hazard:Gfx12Sgpr,gfx11_hazard:Gfx11Hazards,labels:Vec<String>,current_label:String,lds_access_allowed:bool,previous_wmma_dst:Option<reg::RegRef>,pending_barrier:Option<Vec<Transition>>,}
// RDNA4 ISA §5.7.1: bits 15:8 are LOADcnt, bits 7:0 are DScnt.
fn load_ds_wait_imm(load_count:u8,ds_count:u8)->u16 {
 (u16::from(load_count)<<8)|u16::from(ds_count)
}
impl Builder {
 pub fn new(spec:KernelSpec,regs:RegPlan)->Self {let arch=spec.arch;Self{spec,regs,program:Program{arch,instructions:vec![]},ledger:Ledger::default(),lds:Lds::default(),waits:vec![],hazards:vec![],clauses:vec![],barriers:vec![],loop_fixpoints:vec![],hazard:Gfx12Sgpr::default(),gfx11_hazard:Gfx11Hazards::default(),labels:vec!["entry".into()],current_label:"entry".into(),lds_access_allowed:false,previous_wmma_dst:None,pending_barrier:None}}
 pub fn label(&mut self,name:&str)->Result<(),String>{if self.labels.iter().any(|l|l==name){return Err(format!("duplicate label {name}"))}self.labels.push(name.into());self.current_label=name.into();self.program.instructions.push(Instruction::new(format!("{name}:"),vec![],vec![]));Ok(())}
 fn emit_wait(&mut self,c:Counter,n:u8,reason:Reason)->Result<(),String> {let entries=Ledger::wait_instruction(self.spec.arch,&[(c,n,reason.clone())])?;for (_,_,text,_) in entries {let pc_index=self.program.instructions.len();self.program.instructions.push(Instruction::new(text.clone(),vec![],vec![]));self.waits.push(WaitProof{pc_index,insn:text,counter:c,count:n,reason:reason.clone()})}self.ledger.wait(c,n);Ok(())}
 pub fn wait(&mut self,c:Counter,n:u8)->Result<(),String>{self.emit_wait(c,n,Reason::Barrier)}
 fn emit_required(&mut self,mut required:Vec<(Counter,u8,Reason)>)->Result<(),String>{
  if self.spec.arch.gfx12(){
   let load=required.iter().position(|(counter,_,_)|*counter==Counter::Load);
   let ds=required.iter().position(|(counter,_,_)|*counter==Counter::Ds);
   if let (Some(load),Some(ds))=(load,ds){
    let (load_count,load_reason)=(required[load].1,required[load].2.clone());
    let (ds_count,ds_reason)=(required[ds].1,required[ds].2.clone());
    let text=format!("s_wait_loadcnt_dscnt {:#x}",load_ds_wait_imm(load_count,ds_count));
    let pc_index=self.program.instructions.len();
    self.program.instructions.push(Instruction::new(text.clone(),vec![],vec![]));
    self.waits.push(WaitProof{pc_index,insn:text.clone(),counter:Counter::Load,count:load_count,reason:load_reason});
    self.waits.push(WaitProof{pc_index,insn:text,counter:Counter::Ds,count:ds_count,reason:ds_reason});
    self.ledger.wait(Counter::Load,load_count);
    self.ledger.wait(Counter::Ds,ds_count);
    required.retain(|(counter,_,_)|!matches!(counter,Counter::Load|Counter::Ds));
   }
  }
  for (counter,count,reason) in required {self.emit_wait(counter,count,reason)?}
  Ok(())
 }
 pub fn wait_all(&mut self)->Result<(),String>{let required=self.ledger.clone().drain();self.emit_required(required)}
 pub fn push(&mut self,insn:Instruction)->Result<(),String>{insn.validate(self.spec.arch)?;if self.program.instructions.last().is_some_and(|i|i.mnemonic()=="s_endpgm"){return Err("instruction after s_endpgm".into())}
  if insn.mnemonic().starts_with("ds_") && !self.lds_access_allowed {return Err("DS access must carry an LDS slot through ds_load/ds_store".into())}
  if matches!(insn.mnemonic(),"s_clause"|"s_barrier"|"s_barrier_signal"|"s_barrier_wait"|"global_inv"|"buffer_gl0_inv"|"buffer_gl1_inv"){return Err("clauses, barriers and global visibility require their builder methods".into())}
  for r in insn.defs.iter().chain(&insn.uses){self.regs.verify_access(*r,&self.current_label,&self.labels)?}
  self.emit_required(self.ledger.required(&insn))?;
  let mnemonic=insn.mnemonic();let pipe=if matches!(insn.memory,Some(MemoryClass::VmemLoad|MemoryClass::VmemStore)){Pipeline::Vmem}else if insn.memory==Some(MemoryClass::SmemLoad){Pipeline::Smem}else if mnemonic.starts_with('s'){Pipeline::Salu}else if mnemonic.starts_with('v'){Pipeline::Valu}else{Pipeline::Ds};
  if self.spec.arch.gfx12(){let waits=self.hazard.step(pipe,&insn.uses,&insn.defs,false,false);if !waits.is_empty(){let text=format!("s_wait_alu {}",waits.join(" | "));let pc_index=self.program.instructions.len();self.program.instructions.push(Instruction::new(text.clone(),vec![],vec![]));self.hazards.push(HazardProof{pc_index,insn:text,rule:"gfx12 SGPR read-after-write".into()})}}
  if !self.spec.arch.gfx12(){for text in self.gfx11_hazard.step(pipe,mnemonic,&insn.uses,&insn.defs){
   let pc_index=self.program.instructions.len();
   self.program.instructions.push(Instruction::new(text.clone(),vec![],vec![]));
   self.hazards.push(HazardProof{pc_index,insn:text,rule:"gfx11 wave32 trans-use / VMEM SGPR dependence".into()});
  }}
  if mnemonic.starts_with("v_wmma_")||mnemonic.starts_with("v_swmmac_"){
   let dst=*insn.defs.first().ok_or("WMMA destination is missing from instruction defs")?;
   if insn.uses.len()<2{return Err("WMMA A/B operands are missing from instruction uses".into())}
   if self.previous_wmma_dst.is_some_and(|old|insn.uses[..2].iter().any(|source|old.overlaps(*source))||
      (mnemonic.starts_with("v_swmmac_")&&insn.uses.get(2).is_some_and(|index|old.overlaps(*index)))){
    let pc_index=self.program.instructions.len();
    self.program.instructions.push(Instruction::new("v_nop",vec![],vec![]));
    self.hazards.push(HazardProof{pc_index,insn:"v_nop".into(),rule:"WMMA destination feeds next WMMA A/B or SWMMAC index".into()});
   }
   self.previous_wmma_dst=Some(dst);
  }else if pipe==Pipeline::Valu{self.previous_wmma_dst=None}
  self.ledger.record(self.spec.arch,&insn);if mnemonic=="s_endpgm"{self.ledger=Ledger::default()}self.program.instructions.push(insn);Ok(())
 }
 pub fn ds_store(&mut self,slot:usize,insn:Instruction)->Result<(),String>{if insn.memory!=Some(MemoryClass::DsStore)||!insn.mnemonic().starts_with("ds_store"){return Err("ds_store requires an LDS store instruction".into())}let old=self.lds.clone();self.lds.store(slot)?;self.lds_access_allowed=true;let result=self.push(insn);self.lds_access_allowed=false;if result.is_err(){self.lds=old}result}
 pub fn ds_load(&mut self,slot:usize,insn:Instruction)->Result<(),String>{if insn.memory!=Some(MemoryClass::DsLoad)||!insn.mnemonic().starts_with("ds_load"){return Err("ds_load requires an LDS load instruction".into())}let old=self.lds.clone();self.lds.load(slot)?;self.lds_access_allowed=true;let result=self.push(insn);self.lds_access_allowed=false;if result.is_err(){self.lds=old}result}
 pub fn vopd(&mut self,x:vopd::VopdOp,y:vopd::VopdOp)->Result<(),String>{let insn=vopd::packet(self.spec.arch,x,y)?;self.push(insn)}
 pub fn vopd_ff<const DX:u8,const DY:u8,const AX:u8,const AY:u8,const BX:u8,const BY:u8>(&mut self,x:vopd::VopdF32Op<reg::Vp<DX>,vopd::Src0<AX>,reg::Vb<BX>>,y:vopd::VopdF32Op<reg::Vp<DY>,vopd::Src0<AY>,reg::Vb<BY>>)->Result<(),String> where reg::Vp<DX>:reg::OppositeParity<DY>,reg::Vb<AX>:reg::DistinctBanks<AY>,reg::Vb<BX>:reg::DistinctBanks<BY>{self.push(vopd::typed(self.spec.arch,x,y)?)}
 pub fn vopd_ff_shared_src1<const DX:u8,const DY:u8,const AX:u8,const AY:u8>(&mut self,x:vopd::VopdF32Op<reg::Vp<DX>,vopd::Src0<AX>,V<1>>,y:vopd::VopdF32Op<reg::Vp<DY>,vopd::Src0<AY>,V<1>>)->Result<(),String> where reg::Vp<DX>:reg::OppositeParity<DY>,reg::Vb<AX>:reg::DistinctBanks<AY>{self.push(vopd::shared_src1(self.spec.arch,x,y)?)}
 pub fn clause(&mut self,body:impl FnOnce(&mut Builder)->Result<(),String>)->Result<(),String>{
  let mut draft=self.clone();
  let start=draft.program.instructions.len();
  body(&mut draft)?;
  let members=&draft.program.instructions[start..];
  let valid=members.len()>=2&&members.len()<=32
   &&members.iter().all(|i|i.memory==members[0].memory&&i.memory.is_some())
   &&members.iter().enumerate().all(|(i,insn)|!members[..i].iter().any(|prev|
    prev.defs.iter().any(|r|insn.uses.iter().chain(&insn.defs).any(|u|r.overlaps(*u)))));
  if !valid{return Err("clause requires 2..=32 independent instructions of one memory class without waits".into())}
  let len=members.len();
  let class=format!("{:?}",members[0].memory.unwrap());
  draft.program.instructions.insert(start,insn::Sop::Clause(len as u8).encode(self.spec.arch)?);
  draft.clauses.push(plan::ClauseProof{start,len,class});
  *self=draft;
  Ok(())
 }
 pub fn barrier(&mut self,transitions:&[Transition])->Result<(),String>{self.barrier_with_scope(transitions,MemoryScope::LdsOnly)}
 pub fn barrier_signal(&mut self,transitions:&[Transition])->Result<(),String>{
  if !self.spec.arch.gfx12(){return Err("split barriers require gfx12".into())}
  if self.ledger.pending_stores(){self.emit_wait(Counter::Ds,0,Reason::Barrier)?}
  self.lds.barrier_signal(transitions,!self.ledger.pending_stores())?;
  self.pending_barrier=Some(transitions.to_vec());
  self.program.instructions.push(Instruction::new("s_barrier_signal -1",vec![],vec![]));
  Ok(())
 }
 pub fn barrier_wait(&mut self)->Result<(),String>{
  let transitions=self.pending_barrier.take().ok_or("barrier wait without signal")?;
  self.lds.barrier_wait()?;
  let pc_index=self.program.instructions.len();
  self.program.instructions.push(Instruction::new("s_barrier_wait 0xffff",vec![],vec![]));
  self.barriers.push(plan::BarrierProof{pc_index,transitions:transitions.iter().map(|t|format!("{t:?}")).collect()});
  Ok(())
 }
 pub fn barrier_with_scope(&mut self,transitions:&[Transition],scope:MemoryScope)->Result<(),String>{
  if self.spec.arch.gfx12(){self.barrier_signal(transitions)?;self.barrier_wait()?}
  else {
   if self.ledger.pending_stores(){self.emit_wait(Counter::Lgkm,0,Reason::Barrier)?}
   self.lds.barrier(transitions,!self.ledger.pending_stores())?;
   let pc_index=self.program.instructions.len();
   self.program.instructions.push(Instruction::new("s_barrier",vec![],vec![]));
   self.barriers.push(plan::BarrierProof{pc_index,transitions:transitions.iter().map(|t|format!("{t:?}")).collect()});
  }
  if scope==MemoryScope::Workgroup {
   let invalidate=if self.spec.arch.gfx12(){"global_inv scope:SCOPE_SE"}else{"buffer_gl0_inv"};
   self.program.instructions.push(Instruction::new(invalidate,vec![],vec![]));
  }
  Ok(())
 }
 pub fn loop_(&mut self,head:&str,body:impl Fn(&mut Builder)->Result<(),String>)->Result<(),String>{if !self.ledger.is_empty(){return Err("loop entry ledger must be drained".into())}let mut first=self.clone();first.label(head)?;let start=first.program.instructions.len();body(&mut first)?;if !first.ledger.is_empty(){return Err("loop back-edge ledger must be drained".into())}let expected=first.program.instructions[start..].iter().map(|i|i.text.as_str()).collect::<Vec<_>>().join("\n");let mut second=self.clone();second.label(head)?;body(&mut second)?;let actual=second.program.instructions[start..].iter().map(|i|i.text.as_str()).collect::<Vec<_>>().join("\n");if !second.ledger.is_empty()||expected!=actual {return Err("loop wait ledger did not reach fixed point".into())}*self=first;self.loop_fixpoints.push(plan::LoopFixpoint{head:head.into(),iterations:2});Ok(())}
 /// Finish only after every declared lifetime has both endpoints and all live aliases are disjoint.
 pub fn finish(self)->Result<Emitted,String>{if self.pending_barrier.is_some(){return Err("unmatched barrier signal".into())}if self.program.instructions.last().is_none_or(|i|i.mnemonic()!="s_endpgm"){return Err("kernel must end with s_endpgm".into())}if !self.ledger.is_empty(){return Err("outstanding memory operations: wait or end the kernel before finish".into())}self.regs.verify_lifetimes(&self.labels)?;if self.regs.next_free_vgpr()>self.regs.vgpr_budget||self.regs.next_free_sgpr()>self.regs.sgpr_budget {return Err("register plan exceeds budget".into())}let text=emit::assembly(&self.spec,&self.regs,&self.program)?;let hash=format!("{:x}",Sha256::digest(text.as_bytes()));let mut shape=IsaShape {instructions:self.program.instructions.len(),next_free_vgpr:self.regs.next_free_vgpr(),next_free_sgpr:self.regs.next_free_sgpr(),waits:self.waits.len()+self.hazards.len(),barriers:self.barriers.len(),..Default::default()};for i in &self.program.instructions{let name=i.mnemonic();if name.starts_with("v_dual_"){shape.vopd_pairs+=1;shape.valu_slots+=1}else if name.starts_with("v_wmma_"){shape.wmma+=1}else if name.starts_with('v'){shape.valu_slots+=1}if name.starts_with("ds_"){shape.ds+=1}if name.starts_with("buffer_")||name.starts_with("global_"){shape.vmem+=1}}
 let proof=BuilderProof{kernel_id:self.spec.kernel_id,variant:self.spec.variant,arch:self.spec.arch,builder_crate_version:env!("CARGO_PKG_VERSION").into(),builder_git_sha:option_env!("HIPFIRE_BUILDER_GIT_SHA").unwrap_or("unknown").into(),reg_plan:self.regs.ranges,next_free_vgpr:shape.next_free_vgpr,next_free_sgpr:shape.next_free_sgpr,waits:self.waits,hazards:self.hazards,clauses:self.clauses,vopd_pairs:shape.vopd_pairs,lds_slots:self.lds.slots,barriers:self.barriers,loop_fixpoints:self.loop_fixpoints,forbidden_mnemonics_checked:vec!["s_waitcnt (gfx12)".into(),"scratch_*".into()],s_text_sha256:hash};Ok(Emitted{s_text:text,proof,shape})}
}

#[cfg(test)]
mod wait_encoding_tests {
    use super::load_ds_wait_imm;

    #[test]
    fn asymmetric_load_ds_wait_fields() {
        assert_eq!(load_ds_wait_imm(7,3),0x703);
    }
}
