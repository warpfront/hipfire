use crate::{Arch, KernargLayout};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActScale { Row, K128 }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Epi { Set, Add, GateUpSilu, Qkv, Qkvza }
#[derive(Clone, Copy, Debug)]
pub struct Spec { pub arch: Arch, pub act_scale: ActScale, pub epi: Epi }

pub const VGPR_CEILING: u16 = 192;
pub const LDS_BYTES: u32 = 19_456;
pub const SRD_WORD3: u32 = 0x3100_4000;

impl ActScale {
    pub fn name(self) -> &'static str { match self { Self::Row => "row", Self::K128 => "k128" } }
}
impl Epi {
    pub fn name(self) -> &'static str { match self { Self::Set => "set", Self::Add => "add", Self::GateUpSilu => "silu", Self::Qkv => "qkv", Self::Qkvza => "qkvza" } }
    pub fn count(self) -> usize { match self { Self::Set | Self::Add => 1, Self::GateUpSilu => 2, Self::Qkv => 3, Self::Qkvza => 4 } }
}
impl Spec {
    pub fn validate(self) -> Result<(), String> {
        if self.arch != Arch::Gfx1201 { return Err("fp8 GEMM requires gfx1201 wave32".into()) }
        Ok(())
    }
    pub fn symbol(self) -> String { format!("gemm_mq4g256v2_fp8_{}_{}_b1", self.epi.name(), self.act_scale.name()) }
    pub fn module() -> &'static str { "gemm_mq4g256v2_wmma_fp8_gfx12_b1" }
    pub fn variant(self) -> String { format!("ratio-256x128x8-{}.{}", self.act_scale.name(), self.epi.name()) }
    pub fn kernargs() -> KernargLayout {
        let mut layout=KernargLayout::new(96);
        for (name,offset) in ["Wf","Rw","Ew","X8","D","Y0","Y1","Y2","Y3"].into_iter().zip((0..9).map(|i|i*8)) {
            layout=layout.pointer(name,offset);
        }
        for (name,offset) in ["M0","M1","M2","M3","K","N"].into_iter().zip((0..6).map(|i|72+i*4)) {
            layout=layout.hidden(name,offset,4,"by_value");
        }
        layout
    }
}
impl std::str::FromStr for ActScale { type Err=String; fn from_str(s:&str)->Result<Self,String> { match s {"row"=>Ok(Self::Row),"k128"=>Ok(Self::K128),_=>Err(format!("unknown scale layout {s}"))} } }
impl std::str::FromStr for Epi { type Err=String; fn from_str(s:&str)->Result<Self,String> { match s {"set"=>Ok(Self::Set),"add"=>Ok(Self::Add),"silu"=>Ok(Self::GateUpSilu),"qkv"=>Ok(Self::Qkv),"qkvza"=>Ok(Self::Qkvza),_=>Err(format!("unknown fp8 epilogue {s}"))} } }
