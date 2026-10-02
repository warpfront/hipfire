use crate::{Arch, KernargLayout};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActScale { Row, K128 }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Epi { Set, Add, GateUpSilu, GateUpSiluBf16, Qkv, Qkvza,
    /// QKVZA with the GDN preparation fused into the q/k/v rows (see `gdn_epilogue`).
    QkvzaGdn }
#[derive(Clone, Copy, Debug)]
pub struct Spec { pub arch: Arch, pub act_scale: ActScale, pub epi: Epi }

pub const VGPR_CEILING: u16 = 192;
pub const LDS_BYTES: u32 = 19_456;
pub const GDN_KERNARG_BYTES: u32 = 144;

impl ActScale {
    pub fn name(self) -> &'static str { match self { Self::Row => "row", Self::K128 => "k128" } }
}
impl Epi {
    pub fn name(self) -> &'static str { match self { Self::Set => "set", Self::Add => "add", Self::GateUpSilu | Self::GateUpSiluBf16 => "silu", Self::Qkv => "qkv", Self::Qkvza => "qkvza", Self::QkvzaGdn => "qkvzagdn" } }
    pub fn count(self) -> usize { match self { Self::Set | Self::Add => 1, Self::GateUpSilu | Self::GateUpSiluBf16 => 2, Self::Qkv => 3, Self::Qkvza | Self::QkvzaGdn => 4 } }
}
impl Spec {
    pub fn validate(self) -> Result<(), String> {
        if self.arch != Arch::Gfx1201 { return Err("fp8 GEMM requires gfx1201 wave32".into()) }
        if self.epi==Epi::GateUpSiluBf16 && self.act_scale!=ActScale::Row {
            return Err("bf16 h is only supported for row-scaled SiLU".into());
        }
        if self.epi==Epi::QkvzaGdn && self.act_scale!=ActScale::Row {
            return Err("the GDN-fused QKVZA epilogue is only emitted for Row scaling".into());
        }
        Ok(())
    }
    pub fn symbol(self) -> String {
        let base=format!("gemm_mq4g256v2_fp8_{}_{}_b1", self.epi.name(), self.act_scale.name());
        if self.epi==Epi::GateUpSiluBf16 {format!("{base}_bf16")} else {base}
    }
    pub fn module() -> &'static str { "gemm_mq4g256v2_wmma_fp8_gfx12_b1" }
    pub fn variant(self) -> String {
        let base=format!("ratio-256x128x8-{}.{}", self.act_scale.name(), self.epi.name());
        if self.epi==Epi::GateUpSiluBf16 {format!("{base}-bf16")} else {base}
    }
    /// The frozen 96-byte ABI. The GDN-fused QKVZA appends the conv weights,
    /// the persistent conv ring, the FP16 q/k/v bases and q_scale/eps (144 bytes);
    /// its Y0 receives only the raw rows the completion pass reads.
    pub fn kernargs(self) -> KernargLayout {
        let gdn=self.epi==Epi::QkvzaGdn;
        let mut layout=KernargLayout::new(if gdn {GDN_KERNARG_BYTES} else {96});
        for (name,offset) in ["Wf","Rw","Ew","X8","D","Y0","Y1","Y2","Y3"].into_iter().zip((0..9).map(|i|i*8)) {
            layout=layout.pointer(name,offset);
        }
        for (name,offset) in ["M0","M1","M2","M3","K","N"].into_iter().zip((0..6).map(|i|72+i*4)) {
            layout=layout.hidden(name,offset,4,"by_value");
        }
        if gdn {
            for (name,offset) in ["ConvW","ConvState","Q","K16","V"].into_iter().zip((0..5).map(|i|96+i*8)) {
                layout=layout.pointer(name,offset);
            }
            layout=layout.hidden("QScale",136,4,"by_value").hidden("Eps",140,4,"by_value");
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
    let (kernel,_)=rest.split_once("\ts_endpgm").ok_or("F2 LDS certificate missing kernel end")?;
    let (body,epilogue)=kernel.split_once("_epilogue:\n").ok_or("F2 LDS certificate missing epilogue")?;
    // Only the GDN-fused QKVZA epilogue touches LDS; it is certified separately.
    let gdn_end=match epilogue.split_once("_gdn:\n") {
        Some((plain,gdn)) if symbol.contains("_qkvzagdn_") => {
            if plain.lines().any(|l|l.trim().starts_with("ds_")) {return Err(format!("{symbol}: LDS access in the plain epilogue"))}
            if kernel.lines().map(str::trim).filter(|l|l.split_whitespace().nth(1)==Some("v187,")).collect::<Vec<_>>()!=["v_mov_b32_e32 v187, v0"] {
                return Err(format!("{symbol}: thread-id register v187 is redefined"))
            }
            Some(check_gdn_lds(symbol,gdn)?)
        }
        _ => {
            if epilogue.lines().any(|l|l.trim().starts_with("ds_")) {return Err(format!("{symbol}: unproven LDS access in the epilogue"))}
            None
        }
    };
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
        let width=if opcode.ends_with("_b128"){16}else if opcode.ends_with("_b64"){8}else if opcode.ends_with("_b32"){4}else{
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
    Ok(max_end.max(gdn_end.unwrap_or(0)))
}

/// Certify the GDN epilogue's ring accesses (`gdn_epilogue`): every LDS
/// address is one of three registers whose derivations are matched exactly,
/// and the ring-row SGPRs only change through `mod 19` idioms that keep them
/// in [0, 19). Lines exclude labels, waits and issue hints; adjacency below
/// is over the remaining instructions.
fn check_gdn_lds(symbol:&str,text:&str)->Result<u32,String>{
    const RING:u32=19;
    use super::gdn_epilogue::{FIXED_LDS,ROW_BYTES};
    let lines:Vec<&str>=text.lines().map(str::trim).filter(|l|!l.is_empty()&&!l.ends_with(':')&&!l.starts_with("s_delay_alu")&&!l.starts_with("s_wait")).collect();
    let fail=|what:String|Err(format!("{symbol}: GDN LDS certificate: {what}"));
    fn dst(l:&str)->&str{l.split_whitespace().nth(1).map(|t|t.trim_end_matches(',')).unwrap_or("")}
    let writes=|l:&str|!["ds_","buffer_store","s_cmp","s_bitcmp","s_cbranch","s_branch"].iter().any(|p|l.starts_with(p));
    let defs=|reg:&str|->Vec<usize>{lines.iter().enumerate().filter(|(_,l)|writes(l)&&dst(l)==reg).map(|(i,_)|i).collect()};
    let at=|i:usize|lines.get(i).copied().unwrap_or("");
    // Lane and wave terms: lane15 <= 15; hi*32 + rg*256 <= 800; lane*16 + hd*512 <= 1008.
    for (reg,expected) in [
        ("v184",&["v_and_b32_e32 v184, 15, v187"][..]),
        ("v183",&["v_lshrrev_b32_e32 v183, 4, v187","v_and_b32_e32 v183, 1, v183","v_lshlrev_b32_e32 v183, 5, v183","v_add_nc_u32_e32 v183, s97, v183"][..]),
        ("v185",&["v_and_b32_e32 v185, 31, v187","v_lshlrev_b32_e32 v185, 4, v185","v_add_nc_u32_e32 v185, s97, v185"][..]),
        ("s72",&["s_lshr_b32 s72, s88, 1"][..]),
        ("s73",&["s_and_b32 s73, s88, 1"][..]),
    ] {
        if defs(reg).iter().map(|&i|at(i)).collect::<Vec<_>>()!=expected {return fail(format!("{reg} derivation changed"))}
    }
    let before=|reg:&str,i:usize|(0..i).rev().find(|&j|writes(at(j))&&dst(at(j))==reg).map(at);
    for (add,shift) in [("v_add_nc_u32_e32 v183, s97, v183","s_lshl_b32 s97, s88, 8"),("v_add_nc_u32_e32 v185, s97, v185","s_lshl_b32 s97, s72, 9")] {
        let i=lines.iter().position(|l|*l==add).ok_or("missing lane term")?;
        if before("s97",i)!=Some(shift) {return fail(format!("{add} is not preceded by {shift}"))}
    }
    // `x = (x + k) mod 19`: the add, then subtract-and-select once (valid for x + k < 38).
    let mod19=|i:usize,reg:&str|at(i+1)==format!("s_sub_co_i32 s99, {reg}, {RING}")&&at(i+2)==format!("s_cmp_ge_u32 {reg}, {RING}")&&at(i+3)==format!("s_cselect_b32 {reg}, s99, {reg}");
    let in_idiom=|i:usize,reg:&str|at(i)==format!("s_cselect_b32 {reg}, s99, {reg}")&&i>=3&&mod19(i-3,reg);
    for i in defs("s93") {
        let l=at(i);
        if l=="s_mov_b32 s93, 3"||in_idiom(i,"s93")||(l=="s_add_co_i32 s93, s93, 16"&&mod19(i,"s93")) {continue}
        return fail(format!("unproven ring row update `{l}`"))
    }
    for i in defs("s94") {
        let l=at(i);
        let ok=in_idiom(i,"s94")
            ||(l=="s_add_co_i32 s94, s93, 16"&&mod19(i,"s94"))
            ||(l=="s_add_co_i32 s94, s94, 1"&&mod19(i,"s94"))
            ||(l=="s_mov_b32 s94, s94"&&mod19(i,"s94"))
            // + 8*(rg & 1) <= 8, reduced by the idiom that follows.
            ||(l=="s_add_co_i32 s94, s94, s97"&&before("s97",i)==Some("s_lshl_b32 s97, s73, 3")&&at(i+1)=="s_mov_b32 s94, s94"&&mod19(i+1,"s94"));
        if !ok {return fail(format!("unproven ring row update `{l}`"))}
    }
    let (mut max_end,mut accesses)=(0u32,0usize);
    for (i,l) in lines.iter().enumerate() {
        let Some((opcode,operands))=l.split_once(' ') else {continue};
        if !opcode.starts_with("ds_") {continue}
        let (args,offset)=match operands.split_once(" offset:") {Some((a,o))=>(a,o.parse::<u32>().map_err(|_|"invalid GDN LDS offset")?),None=>(operands,0)};
        let addr=if opcode=="ds_store_b128" {args.split(',').next()} else if opcode=="ds_load_b128" {args.split(',').nth(1)} else {return fail(format!("unexpected {opcode}"))}.unwrap_or("").trim();
        let base_max=match (opcode,addr) {
            // Ring row min(r, r - 19) of r = s93 + lane15 <= 33, times the row pitch, plus the channel term.
            ("ds_store_b128","v182")=>{
                let d=defs("v182");
                let j=(0..i).rev().find(|j|d.contains(j)).ok_or("W address undefined")?;
                if !(at(j)==format!("v_mad_u32_u24 v182, v182, {ROW_BYTES:#x}, v183")&&at(j-1)=="v_min_u32_e32 v182, v182, v186"&&at(j-2)==format!("v_subrev_nc_u32_e32 v186, {RING}, v182")&&at(j-3)=="v_add_nc_u32_e32 v182, s93, v184") {
                    return fail("W ring address derivation changed".into())
                }
                (RING-1)*ROW_BYTES+800
            }
            // Halo rows 0..2 through the per-lane head base.
            ("ds_store_b128","v185") if offset<=2*ROW_BYTES => 1008,
            // P rows: ((s94 + k) mod 19) times the row pitch over the head base.
            ("ds_load_b128","v186")=>{
                let row_ok=i>=7&&at(i-1)=="v_add_nc_u32_e32 v186, s97, v185"&&at(i-2)==format!("s_mul_i32 s97, s97, {ROW_BYTES:#x}")&&mod19(i-6,"s97")
                    &&(at(i-6)=="s_mov_b32 s97, s94"||(1..=3).any(|k|at(i-6)==format!("s_add_co_i32 s97, s94, {k}")));
                if !row_ok {return fail(format!("P row address before line `{l}` changed"))}
                (RING-1)*ROW_BYTES+1008
            }
            _=>return fail(format!("unproven LDS base {addr} in `{l}`")),
        };
        let end=offset+base_max+16;
        if end>LDS_BYTES+FIXED_LDS {return fail(format!("`{l}` reaches LDS byte {end}"))}
        max_end=max_end.max(end);accesses+=1;
    }
    if accesses==0 {return fail("no accesses".into())}
    Ok(max_end)
}
impl std::str::FromStr for ActScale { type Err=String; fn from_str(s:&str)->Result<Self,String> { match s {"row"=>Ok(Self::Row),"k128"=>Ok(Self::K128),_=>Err(format!("unknown scale layout {s}"))} } }
impl std::str::FromStr for Epi { type Err=String; fn from_str(s:&str)->Result<Self,String> { match s {"set"=>Ok(Self::Set),"add"=>Ok(Self::Add),"silu"=>Ok(Self::GateUpSilu),"silu-bf16"=>Ok(Self::GateUpSiluBf16),"qkv"=>Ok(Self::Qkv),"qkvza"=>Ok(Self::Qkvza),"qkvzagdn"=>Ok(Self::QkvzaGdn),_=>Err(format!("unknown fp8 epilogue {s}"))} } }

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
