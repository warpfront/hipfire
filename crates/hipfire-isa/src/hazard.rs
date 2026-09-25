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
#[derive(Clone,Debug,Default)] pub struct Gfx11Hazards {last_wmma_dst:Option<RegRef>}
impl Gfx11Hazards { pub fn wmma(&mut self,dst:RegRef,a:RegRef,b:RegRef)->bool {let nop=self.last_wmma_dst.is_some_and(|old|old.overlaps(a)||old.overlaps(b));self.last_wmma_dst=Some(dst);nop} }
