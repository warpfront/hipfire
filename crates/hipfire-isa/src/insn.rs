use crate::{arch::Arch, reg::{Kind, RegRef}};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryClass { VmemLoad, VmemStore, DsLoad, DsStore, SmemLoad, Export }
#[derive(Clone, Debug)]
pub struct Instruction { pub text:String, pub defs:Vec<RegRef>, pub uses:Vec<RegRef>, pub memory:Option<MemoryClass> }
#[derive(Clone, Debug)]
pub struct Program { pub arch:Arch, pub instructions:Vec<Instruction> }
impl Instruction {
    pub fn new(text:impl Into<String>,defs:Vec<RegRef>,uses:Vec<RegRef>)->Self { Self { text:text.into(),defs,uses,memory:None } }
    pub fn memory(mut self,class:MemoryClass)->Self { self.memory=Some(class); self }
    pub fn mnemonic(&self)->&str { self.text.split_whitespace().next().unwrap_or("") }
    pub fn validate(&self,arch:Arch)->Result<(),String> {
        let mnemonic=self.mnemonic();
        arch.check_mnemonic(mnemonic)?;
        if mnemonic.starts_with("scratch_") {return Err("scratch instructions are forbidden".into())}
        fn imm(text:&str)->Result<u32,String> {
            let value=text.trim();
            if let Some(hex)=value.strip_prefix("0x"){u32::from_str_radix(hex,16).map_err(|_|format!("invalid immediate {text}"))}
            else{value.parse().map_err(|_|format!("invalid immediate {text}"))}
        }
        let arg=self.text.split_once(' ').map(|(_,arg)|arg).unwrap_or("");
        let maximum=match mnemonic {
            "s_clause"=>Some(31), "s_wait_kmcnt"=>Some(31),
            "s_wait_loadcnt"|"s_wait_dscnt"|"s_wait_storecnt"=>Some(63),
            _=>None
        };
        if let Some(max)=maximum {
            let value=imm(arg)?;
            if value>max || (mnemonic=="s_clause"&&value==0) {return Err(format!("{mnemonic} immediate {value} exceeds policy/encoding"))}
        }
        if mnemonic=="s_wait_loadcnt_dscnt" {
            let value=imm(arg)?;
            if value & !0x3f3f != 0 {return Err(format!("combined wait {value:#x} exceeds either 6-bit counter field"))}
        }
        if mnemonic.starts_with("buffer_") {
            if let Some((_,offset))=self.text.rsplit_once(" offset:") {
                let value=imm(offset)?;
                if value>arch.buffer_offset_max(){return Err(format!("buffer offset {value} exceeds {} field",arch.name()))}
            }
        }
        if mnemonic.starts_with("ds_") {
            for token in self.text.split_whitespace() {
                if let Some((_,value))=token.split_once("offset0:").or_else(||token.split_once("offset1:")) {
                    if imm(value)?>255 {return Err(format!("DS offset {value} exceeds 8-bit field"))}
                }
            }
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Sop { Clause(u8), WaitLoad(u8), WaitDs(u8), WaitKm(u8), WaitStore(u8), End, Dealloc }
impl Sop { pub fn encode(self,arch:Arch)->Result<Instruction,String> { let t=match self {
    Self::Clause(n) if (2..=32).contains(&n)=>format!("s_clause {}",n-1),
    Self::WaitLoad(n) if n<=63 => if arch.gfx12() { format!("s_wait_loadcnt {n}") } else { format!("s_waitcnt vmcnt({n})") },
    Self::WaitDs(n) if n<=63 => if arch.gfx12() { format!("s_wait_dscnt {n}") } else { format!("s_waitcnt lgkmcnt({n})") },
    Self::WaitKm(n) if arch.gfx12() && n<=31 => format!("s_wait_kmcnt {n}"),
    Self::WaitStore(n) if arch.gfx12() && n<=63 => format!("s_wait_storecnt {n}"),
    Self::End=>"s_endpgm".into(), Self::Dealloc=>"s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)".into(),
    _=>return Err("invalid SOP immediate or unsupported architecture".into()) }; Ok(Instruction::new(t,vec![],vec![])) } }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Vbuffer { LoadB32, LoadB64, StoreB32, StoreB64 }
impl Vbuffer {
    pub fn emit(self,arch:Arch,data:RegRef,voffset:RegRef,srd:RegRef,soffset:Option<RegRef>,offset:u32)->Result<Instruction,String> {
        if offset>arch.buffer_offset_max() { return Err(format!("buffer offset {offset} exceeds {} field",arch.name())) }
        let (name,load,width)=match self { Self::LoadB32=>("buffer_load_b32",true,1),Self::LoadB64=>("buffer_load_b64",true,2),Self::StoreB32=>("buffer_store_b32",false,1),Self::StoreB64=>("buffer_store_b64",false,2) };
        if data.kind!=Kind::V||data.len!=width||voffset.kind!=Kind::V||voffset.len!=1||srd.kind!=Kind::S||srd.len!=4||soffset.is_some_and(|s|s.kind!=Kind::S||s.len!=1) { return Err("invalid buffer operand width/class".into()) }
        let off=soffset.map_or_else(||if arch.gfx12(){"null".into()}else{"0".into()},|r|r.to_string());
        let text=format!("{name} {data}, {voffset}, {srd}, {off} offen{}",if offset==0 {String::new()}else{format!(" offset:{offset}")});
        let mut uses=vec![voffset,srd]; if let Some(s)=soffset { uses.push(s) }
        let defs=if load {vec![data]}else{uses.push(data);vec![]};
        Ok(Instruction::new(text,defs,uses).memory(if load {MemoryClass::VmemLoad}else{MemoryClass::VmemStore}))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Wmma { Iu4, SwmmacIu4 }
impl Wmma { pub fn mnemonic(self,arch:Arch)->Result<&'static str,String> { match self {Self::Iu4=>Ok(arch.wmma_iu4()),Self::SwmmacIu4 if arch.gfx12()=>Ok("v_swmmac_i32_16x16x64_iu4"),_=>Err("SWMMAC requires gfx12".into())} } }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Ds { Load2B64, Load2Stride64B64, Store2Stride64B64, Load2B32 }
impl Ds { pub fn offset(self,offset0:u16,offset1:u16)->Result<String,String> { if offset0>255||offset1>255 { return Err("DS offset exceeds 8 bits".into()) } Ok(format!("{} offset0:{offset0} offset1:{offset1}",match self {Self::Load2B64=>"ds_load_2addr_b64",Self::Load2Stride64B64=>"ds_load_2addr_stride64_b64",Self::Store2Stride64B64=>"ds_store_2addr_stride64_b64",Self::Load2B32=>"ds_load_2addr_b32"})) } }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Vop1 { CvtF32I32, MovB32 }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Vop2 { XorB32, MulF32, FmacF32 }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Vop3 { FmaF32, AddNcU32 }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Vop3p { Dot4I32Iu8 }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Global { LoadB32, LoadB64, StoreB32 }
#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub enum Smem { LoadB256 }

pub const GFX1201_GOLDEN:&[&str]=&[
"s_wait_loadcnt 0x3", "s_wait_dscnt 0x1", "s_wait_loadcnt_dscnt 0x205", "s_wait_kmcnt 0x0", "s_wait_alu depctr_sa_sdst(0)", "s_wait_alu depctr_va_sdst(0)", "s_barrier_signal -1", "s_barrier_wait 0xffff", "global_inv scope:SCOPE_SE", "v_wmma_i32_16x16x32_iu4 v[57:64], v[71:72], v[1:2], 0 neg_lo:[1,1,0]", "v_swmmac_i32_16x16x64_iu4 v[0:7], v[8:9], v[10:13], v14", "v_dual_mul_f32 v83, v97, v81 :: v_dual_mul_f32 v84, v98, v81", "v_dual_fmac_f32 v163, v83, v57 :: v_dual_fmac_f32 v166, v84, v58", "buffer_load_b64 v[0:1], v2, s[4:7], s8 offen", "ds_load_2addr_stride64_b64 v[71:74], v81 offset1:1", "s_clause 0x1", "v_cvt_f32_i32_e32 v57, v57", "s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)", "s_endpgm"];
pub const GFX1100_GOLDEN:&[&str]=&["s_waitcnt vmcnt(3)", "s_waitcnt lgkmcnt(1)", "s_waitcnt_depctr depctr_va_sdst(0)", "s_barrier", "buffer_gl0_inv", "v_wmma_i32_16x16x16_iu4 v[0:7], v[8:9], v[10:11], 0 neg_lo:[1,1,0]", "v_dual_mul_f32 v83, v97, v81 :: v_dual_mul_f32 v84, v98, v82", "v_dual_fmac_f32 v163, v83, v57 :: v_dual_fmac_f32 v166, v84, v58", "buffer_load_b64 v[0:1], v2, s[4:7], s8 offen", "ds_load_2addr_stride64_b64 v[71:74], v81 offset1:1", "s_clause 0x1", "s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)", "s_endpgm"];
