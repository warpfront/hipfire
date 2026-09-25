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

/// Certify the emitted F2 LDS addresses against the launch's dynamic extent.
/// The five hoisted address registers have fixed lane/wave bounds for a
/// 256-thread workgroup; reject an address-map change rather than trusting an
/// obsolete bound. Offsets are read from the emitted instructions themselves.
pub fn check_lds_access(source:&str,symbol:&str,launch_dynamic:u32)->Result<u32,String>{
    if launch_dynamic!=LDS_BYTES {return Err(format!("{symbol}: expected {LDS_BYTES} launch LDS bytes, got {launch_dynamic}"))}
    let marker=format!("\n{symbol}:\n");
    let (_,rest)=source.split_once(&marker).ok_or("F2 LDS certificate missing symbol")?;
    let (body,_)=rest.split_once("\ts_endpgm").ok_or("F2 LDS certificate missing kernel end")?;
    let (body,_)=body.split_once("_epilogue:\n").ok_or("F2 LDS certificate missing epilogue")?;
    for (reg,expected) in [
        (178,&["v_lshrrev_b32_e32 v178, 4, v188","v_lshlrev_b32_e32 v178, 8, v178","v_add_nc_u32_e32 v178, v178, v171","v_add_nc_u32_e32 v178, v178, v171"][..]),
        (179,&["v_and_b32_e32 v179, 31, v0","v_lshlrev_b32_e32 v179, 3, v179","v_add_nc_u32_e32 v179, s72, v179"][..]),
        (180,&["v_and_b32_e32 v180, 31, v0","v_lshrrev_b32_e32 v180, 4, v180","v_lshlrev_b32_e32 v180, 5, v180","v_add_nc_u32_e32 v180, s72, v180"][..]),
        (181,&["v_and_b32_e32 v181, 15, v0","v_lshlrev_b32_e32 v181, 2, v181","v_add_nc_u32_e32 v181, s72, v181"][..]),
        (182,&["v_lshlrev_b32_e32 v182, 2, v0"][..]),
    ] {
        let prefix=format!("v{reg},");
        let actual=body.lines().map(str::trim).filter(|line|
            line.starts_with("v_") && line.split_whitespace().nth(1)==Some(prefix.as_str()))
            .collect::<Vec<_>>();
        if actual!=expected {return Err(format!("{symbol}: LDS address v{reg} derivation changed"))}
    }
    for formula in [
        "v_readfirstlane_b32 s87, v0",
        "s_lshr_b32 s87, s87, 5",
        "s_and_b32 s88, s87, 3",
        "s_lshr_b32 s89, s87, 2",
        "s_lshl_b32 s72, s89, 10",
        "s_lshl_b32 s72, s88, 8",
        "s_lshl_b32 s72, s89, 8",
    ] {
        if !body.lines().any(|line|line.trim()==formula) {
            return Err(format!("{symbol}: LDS wave address formula changed: {formula}"))
        }
    }
    let mut max_end=0;
    let mut mask_ready=false;
    let mut masked=false;
    let mut accesses=0;
    for line in body.lines().map(str::trim) {
        if line=="v_cmp_gt_u32_e64 s72, 0x80, v187" {mask_ready=true}
        if line=="s_mov_b32 exec_lo, s72" {
            if !mask_ready {return Err(format!("{symbol}: unproven LDS token mask"))}
            masked=true;
        }
        if line=="s_mov_b32 exec_lo, s74" {masked=false;mask_ready=false}
        let Some((opcode,operands))=line.split_once(' ') else {continue};
        if !opcode.starts_with("ds_") {continue}
        let width=if opcode.ends_with("_b64"){8}else if opcode.ends_with("_b32"){4}else{
            return Err(format!("{symbol}: unrecognized LDS access {opcode}"))
        };
        let (args,offset)=if let Some((args,imm))=operands.split_once(" offset:"){
            (args,imm.parse::<u32>().map_err(|_|"invalid F2 LDS offset")?)
        }else{(operands,0)};
        let addr=if opcode.starts_with("ds_store_"){args.split(',').next()}else{args.split(',').nth(1)}
            .ok_or("F2 LDS address operand missing")?.trim();
        let (base_max,lo,hi)=match addr {
            "v178"=>(6008,0,16384), // (tid>>5)*256 + ((tid>>1)&15)*8 + (tid&1)*4096
            "v179"=>(1272,0,16384), // (tid&31)*8 + ((tid>>7)&1)*1024
            "v180"=>(800,16384,18432), // ((tid&31)>>4)*32 + ((tid>>5)&3)*256
            "v181"=>(316,18432,LDS_BYTES), // (tid&15)*4 + ((tid>>7)&1)*256
            "v182" if offset<18432=>(1020,16384,18432), // tid*4, ratio planes
            "v182" if masked=>(508,18432,LDS_BYTES), // tid<128, D planes
            _=>return Err(format!("{symbol}: unproven LDS base {addr} at offset {offset}")),
        };
        let end=offset+base_max+width;
        if offset<lo || end>hi {return Err(format!("{symbol}: {opcode} {addr} touches LDS byte {end} outside [{lo},{hi})"))}
        max_end=max_end.max(end);accesses+=1;
    }
    if accesses==0 {return Err(format!("{symbol}: no LDS accesses certified"))}
    Ok(max_end)
}
impl std::str::FromStr for ActScale { type Err=String; fn from_str(s:&str)->Result<Self,String> { match s {"row"=>Ok(Self::Row),"k128"=>Ok(Self::K128),_=>Err(format!("unknown scale layout {s}"))} } }
impl std::str::FromStr for Epi { type Err=String; fn from_str(s:&str)->Result<Self,String> { match s {"set"=>Ok(Self::Set),"add"=>Ok(Self::Add),"silu"=>Ok(Self::GateUpSilu),"qkv"=>Ok(Self::Qkv),"qkvza"=>Ok(Self::Qkvza),_=>Err(format!("unknown fp8 epilogue {s}"))} } }

#[cfg(test)]
mod tests {
    use super::{ActScale,Epi,Spec,check_lds_access};
    use crate::Arch;

    #[test]
    fn emitted_lds_extent_rejects_overflow_or_short_launch() {
        for (scale,end) in [(ActScale::Row,18432),(ActScale::K128,19456)] {
            let spec=Spec{arch:Arch::Gfx1201,act_scale:scale,epi:Epi::GateUpSilu};
            let source=super::super::emit(spec).unwrap().s_text;
            assert_eq!(check_lds_access(&source,&spec.symbol(),19456).unwrap(),end);
            assert!(check_lds_access(&source,&spec.symbol(),19455).is_err());
            let mutant=source.replacen("ds_load_b64 v[160:161], v179 offset:8192",
                "ds_load_b64 v[160:161], v179 offset:18000",1);
            assert_ne!(mutant,source);
            assert!(check_lds_access(&mutant,&spec.symbol(),19456).is_err());
        }
    }
}
