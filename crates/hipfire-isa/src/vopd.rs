use crate::{arch::Arch, insn::Instruction, reg::{DistinctBanks,OppositeParity,RegRef,V,Vb,Vp,S}};
#[derive(Clone,Copy,Debug,PartialEq,Eq)] pub enum VopdF32 { Add,Sub,Subrev,Mul,Fmac }
impl VopdF32 { fn name(self)->&'static str {match self {Self::Add=>"v_dual_add_f32",Self::Sub=>"v_dual_sub_f32",Self::Subrev=>"v_dual_subrev_f32",Self::Mul=>"v_dual_mul_f32",Self::Fmac=>"v_dual_fmac_f32"}} }
#[derive(Clone,Copy,Debug,PartialEq,Eq)] pub enum Src0<const B:u8> { V(Vb<B>),S(S<1>),Lit(u32),Inline(i8) }
#[derive(Clone,Copy,Debug)] pub struct VopdF32Op<D,A,B> { pub op:VopdF32,pub dst:D,pub src0:A,pub src1:B }
#[derive(Clone,Copy,Debug,PartialEq,Eq)] pub enum Operand { V(u8),S(u8),Lit(u32),Inline(i8) }
impl Operand { fn text(self)->String {match self {Self::V(n)=>format!("v{n}"),Self::S(n)=>format!("s{n}"),Self::Lit(n)=>format!("0x{n:08x}"),Self::Inline(n)=>n.to_string()}} fn reg(self)->Option<RegRef> {match self {Self::V(n)=>Some(V::<1>(n).reg()),Self::S(n)=>Some(S::<1>(n).reg()),_=>None}} }
#[derive(Clone,Copy,Debug)] pub struct VopdOp {pub op:VopdF32,pub dst:u8,pub src0:Operand,pub src1:u8}
impl VopdOp {fn text(self)->String {format!("{} v{}, {}, v{}",self.op.name(),self.dst,self.src0.text(),self.src1)} }
pub fn validate_pair(arch:Arch,x:VopdOp,y:VopdOp)->Result<(),String> {
    if x.dst%2==y.dst%2 {return Err("VOPD destination parity collision".into())}
    if let (Operand::V(a),Operand::V(b))=(x.src0,y.src0) {if a%4==b%4 {return Err("VOPD src0 bank collision".into())}}
    if x.src1%4==y.src1%4 && !(arch.gfx12()&&x.src1==y.src1) {return Err("VOPD src1 bank collision".into())}
    let literals=[x.src0,y.src0].iter().filter_map(|s|if let Operand::Lit(n)=s {Some(*n)}else{None}).collect::<Vec<_>>();
    if literals.len()>1 && literals[0]!=literals[1] {return Err("VOPD supports only one shared literal".into())}
    Ok(())
}
pub fn packet(arch:Arch,x:VopdOp,y:VopdOp)->Result<Instruction,String> {validate_pair(arch,x,y)?;let mut uses=vec![V::<1>(x.src1).reg(),V::<1>(y.src1).reg()];for op in [x.src0,y.src0] {if let Some(r)=op.reg(){uses.push(r)}}; if x.op==VopdF32::Fmac {uses.push(V::<1>(x.dst).reg())} if y.op==VopdF32::Fmac {uses.push(V::<1>(y.dst).reg())} Ok(Instruction::new(format!("{} :: {}",x.text(),y.text()),vec![V::<1>(x.dst).reg(),V::<1>(y.dst).reg()],uses)) }
pub fn typed<const DX:u8,const DY:u8,const AX:u8,const AY:u8,const BX:u8,const BY:u8>(arch:Arch,x:VopdF32Op<Vp<DX>,Src0<AX>,Vb<BX>>,y:VopdF32Op<Vp<DY>,Src0<AY>,Vb<BY>>)->Result<Instruction,String> where Vp<DX>:OppositeParity<DY>,Vb<AX>:DistinctBanks<AY>,Vb<BX>:DistinctBanks<BY> {
    fn convert<const B:u8>(s:Src0<B>)->Operand {match s {Src0::V(v)=>Operand::V(v.0),Src0::S(s)=>Operand::S(s.0),Src0::Lit(n)=>Operand::Lit(n),Src0::Inline(n)=>Operand::Inline(n)}}
    if x.dst.0%2!=DX||y.dst.0%2!=DY||matches!(x.src0,Src0::V(v) if v.0%4!=AX)||matches!(y.src0,Src0::V(v) if v.0%4!=AY)||x.src1.0%4!=BX||y.src1.0%4!=BY {return Err("typed VOPD register carries wrong bank/parity".into())}
    packet(arch,VopdOp{op:x.op,dst:x.dst.0,src0:convert(x.src0),src1:x.src1.0},VopdOp{op:y.op,dst:y.dst.0,src0:convert(y.src0),src1:y.src1.0})
}
pub fn shared_src1<const DX:u8,const DY:u8,const AX:u8,const AY:u8>(arch:Arch,x:VopdF32Op<Vp<DX>,Src0<AX>,V<1>>,y:VopdF32Op<Vp<DY>,Src0<AY>,V<1>>)->Result<Instruction,String> where Vp<DX>:OppositeParity<DY>,Vb<AX>:DistinctBanks<AY> {
    if !arch.gfx12()||x.src1!=y.src1||x.dst.0%2!=DX||y.dst.0%2!=DY {return Err("shared-src1 packet requires gfx12, identical src1 and correct parity".into())}
    fn cv<const B:u8>(s:Src0<B>)->Operand {match s { Src0::V(Vb(n))=>Operand::V(n),Src0::S(S(n))=>Operand::S(n),Src0::Lit(n)=>Operand::Lit(n),Src0::Inline(n)=>Operand::Inline(n) }}
    packet(arch,VopdOp{op:x.op,dst:x.dst.0,src0:cv(x.src0),src1:x.src1.0},VopdOp{op:y.op,dst:y.dst.0,src0:cv(y.src0),src1:y.src1.0})
}
