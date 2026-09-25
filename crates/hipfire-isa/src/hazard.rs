use crate::reg::{Kind,RegRef};
use serde::Serialize;
#[derive(Clone,Copy,Debug,PartialEq,Eq)] pub enum Pipeline { Salu,Valu,Vmem,Smem,Ds }
#[derive(Clone,Debug,Serialize)] pub struct HazardProof { pub pc_index:usize,pub insn:String,pub rule:String }
#[derive(Clone,Debug)] pub struct Gfx12Sgpr { tracked:[bool;64],salu:[bool;128],valu:[bool;128],vcc_salu:bool,vcc_valu:bool }
impl Default for Gfx12Sgpr { fn default()->Self {Self {tracked:[false;64],salu:[false;128],valu:[false;128],vcc_salu:false,vcc_valu:false}} }
impl Gfx12Sgpr {
    pub fn clear_on_memory(&mut self) { self.salu.fill(false);self.valu.fill(false);self.vcc_salu=false;self.vcc_valu=false; }
    pub fn step(&mut self,pipe:Pipeline,uses:&[RegRef],defs:&[RegRef],vcc_use:bool,vcc_def:bool)->Vec<&'static str> {
        if matches!(pipe,Pipeline::Vmem|Pipeline::Smem) {self.clear_on_memory();return vec![]}
        if !matches!(pipe,Pipeline::Salu|Pipeline::Valu) {return vec![]}
        let mut sa=false;let mut va=false;let mut vcc=false;let mut salu_read_tracked=false;
        for r in uses { if r.kind!=Kind::S {continue} for n in r.base as usize..r.base as usize+r.len as usize {if self.tracked[n/2] {sa|=self.salu[n];if pipe==Pipeline::Valu {va|=self.valu[n]}else{salu_read_tracked=true}}} }
        if vcc_use {sa|=self.vcc_salu;if pipe==Pipeline::Valu {vcc|=self.vcc_valu}else{self.vcc_valu=false}}
        let mut waits=Vec::new(); if sa {waits.push("depctr_sa_sdst(0)");self.salu.fill(false);self.vcc_salu=false} if va {waits.push("depctr_va_sdst(0)");self.valu.fill(false)} if vcc {waits.push("depctr_va_vcc(0)");self.vcc_valu=false}
        if salu_read_tracked {self.valu.fill(false)}
        for r in uses {if r.kind==Kind::S && pipe==Pipeline::Valu {for n in r.base as usize..r.base as usize+r.len as usize {self.tracked[n/2]=true}}}
        for r in defs {if r.kind==Kind::S {for n in r.base as usize..r.base as usize+r.len as usize {if self.tracked[n/2] {if pipe==Pipeline::Salu {self.salu[n]=true}else{self.valu[n]=true}}}}}
        if vcc_def {self.vcc_salu=pipe==Pipeline::Salu;self.vcc_valu=pipe==Pipeline::Valu} waits
    }
}
/// gfx11 wave32 hazards. The mask-write and partial-forwarding hazards in LLVM
/// apply only to wave64, which this builder does not emit.
#[derive(Clone,Debug,Default)]
pub struct Gfx11Hazards {
    trans_defs: Vec<(RegRef,u8,u8)>,
    valu_sgpr_defs: Vec<(RegRef,u8)>,
}
impl Gfx11Hazards {
    pub fn step(&mut self,pipe:Pipeline,mnemonic:&str,uses:&[RegRef],defs:&[RegRef])->Vec<String>{
        let mut waits=Vec::new();
        let trans=pipe==Pipeline::Valu && ["v_rcp_","v_sqrt_","v_rsq_","v_sin_","v_cos_","v_exp_","v_log_"].iter().any(|prefix|mnemonic.starts_with(prefix));
        if matches!(pipe,Pipeline::Vmem|Pipeline::Ds) {
            self.trans_defs.clear();
        }
        if pipe==Pipeline::Vmem {
            let needed=self.valu_sgpr_defs.iter().filter(|(def,_)|uses.iter().any(|r|r.overlaps(*def)))
                .map(|(_,age)|5u8.saturating_sub(*age)).max().unwrap_or(0);
            if needed>0 {
                waits.push(format!("s_nop {}",needed-1));
                self.valu_sgpr_defs.clear();
            }
        }
        if pipe==Pipeline::Valu && self.trans_defs.iter().any(|(def,valu_age,trans_age)|
            *valu_age<=5 && *trans_age<=1 && uses.iter().any(|r|r.overlaps(*def))) {
            waits.push("s_waitcnt_depctr depctr_va_vdst(0)".into());
            self.trans_defs.clear();
        }
        for (_,age) in &mut self.valu_sgpr_defs { *age=age.saturating_add(1) }
        self.valu_sgpr_defs.retain(|(_,age)|*age<5);
        if pipe==Pipeline::Valu {
            for (_,valu_age,trans_age) in &mut self.trans_defs {
                *valu_age=valu_age.saturating_add(1);
                if trans {*trans_age=trans_age.saturating_add(1)}
            }
            self.trans_defs.retain(|(_,valu_age,trans_age)|*valu_age<=5&&*trans_age<=1);
            self.valu_sgpr_defs.extend(defs.iter().filter(|r|r.kind==Kind::S).map(|r|(*r,0)));
            if trans {self.trans_defs.extend(defs.iter().filter(|r|r.kind==Kind::V).map(|r|(*r,0,0)))}
        }
        waits
    }
}
/// gfx11/gfx12 `s_delay_alu` scheduling hints. The hardware interlocks VALU
/// dependencies, but a dependent VALU issued without the hint stalls the SIMD's
/// VALU pipeline instead of yielding to another wave. Before each VALU
/// instruction that reads a VGPR written by one of the last four VALU
/// instructions (or last three transcendentals) this names the nearest such
/// producer, as LLVM's AMDGPUInsertDelayAlu does. Hints never affect results.
#[derive(Clone,Debug,Default)]
pub struct DelayAlu { recent: Vec<(Vec<RegRef>,bool)> }
impl DelayAlu {
    const TRANS: [&'static str; 7] = ["v_rcp_","v_sqrt_","v_rsq_","v_sin_","v_cos_","v_exp_","v_log_"];
    /// A branch target may be reached from another stream: forget producers.
    pub fn label(&mut self) { self.recent.clear() }
    /// Hint text for a VALU instruction (None when no recent producer), then
    /// record the instruction as the newest producer.
    pub fn step(&mut self,mnemonic:&str,uses:&[RegRef],defs:&[RegRef])->Option<String> {
        let (mut valu,mut trans,mut trans_seen)=(None,None,0usize);
        for (distance,(written,is_trans)) in self.recent.iter().rev().enumerate().map(|(i,r)|(i+1,r)) {
            if *is_trans {trans_seen+=1}
            if !written.iter().any(|w|uses.iter().any(|u|u.kind==Kind::V&&w.overlaps(*u))) {continue}
            if *is_trans&&trans_seen<=3 {trans.get_or_insert(trans_seen);} else if !*is_trans {valu.get_or_insert(distance);}
        }
        let trans_insn=Self::TRANS.iter().any(|p|mnemonic.starts_with(p));
        self.recent.push((defs.iter().copied().filter(|r|r.kind==Kind::V).collect(),trans_insn));
        if self.recent.len()>4 {self.recent.remove(0);}
        match (valu,trans) {
            (None,None)=>None,
            (Some(v),None)=>Some(format!("s_delay_alu instid0(VALU_DEP_{v})")),
            (None,Some(t))=>Some(format!("s_delay_alu instid0(TRANS32_DEP_{t})")),
            (Some(v),Some(t))=>Some(format!("s_delay_alu instid0(VALU_DEP_{v}) | instid1(TRANS32_DEP_{t})")),
        }
    }
}
