//! In-order wave32 interpreter. Wait counters are no-ops: loads complete before
//! the next instruction. Workgroup barriers still rendezvous distinct waves.
mod alu;
mod memory;
pub mod convert;
pub mod trans;
pub mod mma;
mod wmma;
use crate::{inst::{Abi, Wave}, operand::{Half, ImmField, InlineConst, Operand, Special}, reg::Kind, Arch, Inst, Kernel, Program};

pub type Result<T, E = String> = std::result::Result<T, E>;
#[derive(Clone, Debug)]
pub struct WriteSet { bits: Vec<u64>, bytes: usize }
impl WriteSet {
    fn new(bytes:usize)->Self {Self{bits:vec![0;bytes.div_ceil(64)],bytes}}
    pub fn contains(&self,byte:usize)->bool {byte<self.bytes && self.bits[byte/64]&(1u64<<(byte%64))!=0}
    pub fn count(&self)->usize {self.bits.iter().map(|b|b.count_ones()as usize).sum()}
    fn mark(&mut self,start:usize,len:usize) {
        let mut pos=start;let end=start+len;
        while pos<end {let n=(64-pos%64).min(end-pos);let mask=if n==64{u64::MAX}else{((1u64<<n)-1)<<(pos%64)};self.bits[pos/64]|=mask;pos+=n;}
    }
}
#[derive(Clone, Debug)]
pub struct Region { pub name: String, pub base: u64, pub bytes: Vec<u8>, pub written: WriteSet }
#[derive(Clone, Debug, Default)]
pub struct Memory { pub regions: Vec<Region> }
impl Memory {
    pub fn map(&mut self, name: String, base: u64, bytes: Vec<u8>) -> Result<()> {
        let end = base.checked_add(bytes.len() as u64).ok_or("region address overflow")?;
        if self.regions.iter().any(|r| base < r.base + r.bytes.len() as u64 && r.base < end) { return Err("overlapping captured regions".into()); }
        self.regions.push(Region { name, base, written: WriteSet::new(bytes.len()), bytes }); Ok(())
    }
    fn locate(&self, addr: u64, n: usize) -> Result<(usize, usize)> {
        self.regions.iter().enumerate().find_map(|(i,r)| { let off = addr.checked_sub(r.base)?; let end = off.checked_add(n as u64)?; (end <= r.bytes.len() as u64).then_some((i,off as usize)) })
            .ok_or_else(|| format!("unmapped memory access {addr:#x} size {n}"))
    }
    pub fn read(&self, addr: u64, n: usize) -> Result<&[u8]> { let (i,o)=self.locate(addr,n)?; Ok(&self.regions[i].bytes[o..o+n]) }
    pub fn write(&mut self, addr: u64, bytes: &[u8]) -> Result<()> { let (i,o)=self.locate(addr,bytes.len())?; let r=&mut self.regions[i]; r.bytes[o..o+bytes.len()].copy_from_slice(bytes); r.written.mark(o,bytes.len()); Ok(()) }
}

pub struct Launch { pub kernarg_ptr: u64, pub workgroup: [u32;3], pub block: [u32;3], pub dynamic_lds: u32 }
struct State { s: [u32;106], t: [u32;16], v: Box<[[u32;32];256]>, exec: u32, vcc: u32, scc: bool, m0: u32, pc: usize, ended: bool, barrier: Option<u64>, signals: u64 }
impl State {
    fn new(arch: Arch, kernel: &Kernel, launch: &Launch, wave: usize) -> Result<Self> {
        if kernel.wave != Wave::Wave32 { return Err("emulator requires wave32".into()); }
        let Abi::Hsa { descriptor:d, .. } = &kernel.abi else { return Err("descriptor-driven HSA ABI required".into()); };
        if d.private_segment_fixed_size != 0 { return Err("private scratch allocation unsupported".into()); }
        let mode=d.compute_pgm_rsrc1.0;
        if (mode>>12)&255!=0xf0 || mode&(1<<26)!=0 {
            return Err("only RNE with preserved denormals and FP16_OVFL=0 is modeled".into());
        }
        if arch==Arch::Gfx1151 && mode&((1<<23)|(1<<21))!=((1<<23)|(1<<21)) {
            return Err("gfx1151 emulator requires IEEE_MODE and DX10_CLAMP".into());
        }
        let mut st=Self { s:[0;106],t:[0;16],v:Box::new([[0;32];256]),exec:0,vcc:0,scc:false,m0:0,pc:0,ended:false,barrier:None,signals:0 };
        let props=d.kernel_code_properties.0;
        let mut sg=0;
        for (bit,n) in [(0,4),(1,2),(2,2),(3,2),(4,2),(5,2),(6,1)] {
            if props & (1<<bit) == 0 { continue; }
            if bit != 3 { return Err(format!("unsupported descriptor user SGPR property {bit}")); }
            st.s[sg]=launch.kernarg_ptr as u32; st.s[sg+1]=(launch.kernarg_ptr>>32) as u32; sg+=n;
        }
        let rs=d.compute_pgm_rsrc2.0;
        if sg != ((rs>>1)&31) as usize { return Err("descriptor user SGPR count disagrees with enabled properties".into()); }
        if arch == Arch::Gfx1201 {
            st.t[9]=launch.workgroup[0];
            st.t[7]=if rs&(1<<8)!=0 {launch.workgroup[1]&0xffff}else{0}
                |if rs&(1<<9)!=0 {(launch.workgroup[2]&0xffff)<<16}else{0};
            st.t[8]=((wave as u32&31)<<25)|if rs&((1<<8)|(1<<9))!=0 {1<<30}else{0};
        } else {
            for (axis,bit) in [7,8,9].into_iter().enumerate() { if rs&(1<<bit)!=0 {st.s[sg]=launch.workgroup[axis];sg+=1;} }
            if rs&(1<<10)!=0 { st.s[sg]=launch.block[0]*launch.block[1]*launch.block[2]; }
        }
        let preload=d.kernarg_preload.0;
        if preload != 0 { return Err("kernarg preload ABI unsupported".into()); }
        let workitem_mode=(rs>>11)&3;
        if workitem_mode==3 {return Err("undefined ENABLE_VGPR_WORKITEM_ID=3".into());}
        let threads=launch.block[0].checked_mul(launch.block[1]).and_then(|n|n.checked_mul(launch.block[2])).ok_or("block size overflow")? as usize;
        for lane in 0..32 { let id=wave*32+lane; if id>=threads {continue;} st.exec |= 1<<lane;
            let x=id as u32 % launch.block[0]; let y=id as u32 / launch.block[0] % launch.block[1]; let z=id as u32 / (launch.block[0]*launch.block[1]);
            st.v[0][lane]=x|(if workitem_mode>0 {y<<10}else{0})|(if workitem_mode>1 {z<<20}else{0});
        }
        Ok(st)
    }
    fn read(&self, op:&Operand,lane:usize,word:usize)->Result<u32> {
        Ok(match op {
            Operand::Reg(r)=> { let n=usize::from(r.base)+word; if word>=usize::from(r.len) {return Err("register component exceeds operand width".into());} match r.kind {Kind::S=>self.s[n],Kind::V=>self.v[n][lane],Kind::Ttmp=>self.t[n]} },
            Operand::Half(r,h)=> { let bits=self.read(&Operand::Reg(*r),lane,0)?; (bits>>if *h==Half::Hi {16}else{0})&0xffff },
            Operand::Special(s)=>match s {Special::Exec|Special::ExecLo=>self.exec, Special::Vcc|Special::VccLo=>self.vcc,Special::ExecHi|Special::VccHi|Special::Null=>0,Special::Scc=>u32::from(self.scc),Special::M0=>self.m0,Special::Ttmp(n)=>self.t[usize::from(*n)],_=>return Err(format!("special source {s:?} unsupported"))},
            Operand::Inline(InlineConst::Integer(n))=>*n as i32 as u32,Operand::Inline(InlineConst::FloatBits(n))|Operand::Literal(n)=>*n,Operand::Inline(InlineConst::InvTwoPi)=>0x3e22f983,
            Operand::Imm(i)=>match i {ImmField::Sopp(n)|ImmField::Sopk(n)=>*n as i32 as u32,ImmField::Unsigned(n)=>*n,ImmField::DsOffset(n)=>u32::from(*n),ImmField::DsOffset0(n)|ImmField::DsOffset1(n)=>u32::from(*n),ImmField::VmemOffset(n)|ImmField::SmemOffset(n)|ImmField::SmemDisplacement(n)=>*n as u32},
            _=>return Err(format!("not a value operand {op:?}")),
        })
    }
    fn read64(&self,op:&Operand,lane:usize)->Result<u64> {Ok(u64::from(self.read(op,lane,0)?)|u64::from(self.read(op,lane,1)?)<<32)}
    fn put(&mut self,op:&Operand,lane:usize,word:usize,value:u32)->Result<()> {
        match op {Operand::Reg(r)=>{let n=usize::from(r.base)+word;match r.kind {Kind::S=>self.s[n]=value,Kind::V=>self.v[n][lane]=value,Kind::Ttmp=>self.t[n]=value}},
            Operand::Half(r,h)=>{let old=self.read(&Operand::Reg(*r),lane,0)?;let shift=if *h==Half::Hi {16}else{0};self.put(&Operand::Reg(*r),lane,0,(old&!(0xffff<<shift))|((value&0xffff)<<shift))?;},
            Operand::Special(s)=>match s {Special::Exec|Special::ExecLo=>self.exec=value,Special::Vcc|Special::VccLo=>self.vcc=value,Special::Scc=>self.scc=value!=0,Special::M0=>self.m0=value,Special::Null=>(),_=>return Err(format!("special destination {s:?} unsupported"))},
            _=>return Err("non-register destination".into())} Ok(())
    }
}

/// Name-keyed semantic side table, deliberately separate from encoding tables.
pub fn semantic_class(name:&str)->Option<&'static str> {
    static TABLE:std::sync::LazyLock<std::collections::HashMap<&'static str,&'static str>>=std::sync::LazyLock::new(|| {
        include_str!("../../isa/semantics.tbl").lines().filter(|l|!l.starts_with('#')).filter_map(|l|l.split_once('\t')).collect()
    });
    TABLE.get(name).copied()
}
#[derive(Clone,Copy)]
enum Class { Wait, Control, Memory, Alu }
struct Step<'a> { inst:&'a Inst, name:&'static str, class:Class }
/// Decode/name/semantic lookup is compiled once, not repeated inside hot loops
/// or for each workgroup of a real-shape replay.
pub struct Emulator<'a> { kernel:&'a Kernel, arch:Arch, steps:Vec<Step<'a>> }
impl<'a> Emulator<'a> {
    pub fn new(program:&'a Program,symbol:&str)->Result<Self> {
        let kernel=program.kernels.iter().find(|k|k.symbol.0==symbol).ok_or("snapshot symbol absent in Program")?;
        let steps=kernel.body.layout.iter().map(|id| {
            let inst=kernel.body.insts.get(*id).ok_or("missing instruction")?;
            let name=crate::isa::lookup(program.target.arch,inst.op,inst.form).ok_or("unknown decoded opcode")?.name;
            let class=match semantic_class(name).ok_or_else(||format!("no semantics for {name}"))? {
                "wait"=>Class::Wait,"control"=>Class::Control,"memory"=>Class::Memory,"alu"=>Class::Alu,
                class=>return Err(format!("unsupported semantics class {class} for {name}")),
            };
            Ok(Step{inst,name,class})
        }).collect::<Result<Vec<_>>>()?;
        Ok(Self{kernel,arch:program.target.arch,steps})
    }
    pub fn run_workgroup(&self,launch:&Launch,mem:&mut Memory)->Result<()> {
        let kernel=self.kernel;let symbol=&kernel.symbol.0;
        let Abi::Hsa {descriptor:d,..}=&kernel.abi else {return Err("HSA descriptor required".into());};
        let n=launch.block[0].checked_mul(launch.block[1]).and_then(|n|n.checked_mul(launch.block[2])).ok_or("block size overflow")? as usize;
        if n==0 || n>1024 {return Err("workgroup must contain 1..1024 threads".into());}
        let mut states=(0..n.div_ceil(32)).map(|w|State::new(self.arch,kernel,launch,w)).collect::<Result<Vec<_>>>()?;
        let mut lds=vec![0u8; d.group_segment_fixed_size.checked_add(launch.dynamic_lds).ok_or("LDS size overflow")? as usize];
        let mut executed=0u64;
        while states.iter().any(|s|!s.ended) {
            let mut progress=false;
            for (wave,st) in states.iter_mut().enumerate() {
                if st.ended || st.barrier.is_some() {continue;}
                let step=self.steps.get(st.pc).ok_or("PC ran past kernel end")?;
                let inst=step.inst;let name=step.name;
                st.pc+=1;progress=true;executed+=1;
                if executed>100_000_000 {return Err("instruction limit exceeded (possible divergent loop)".into());}
                let result=match step.class {
                    Class::Wait=>Ok(()),Class::Control=>control(name,inst,st,kernel),
                    Class::Memory=>memory::execute(name,inst,st,mem,&mut lds),
                    Class::Alu=>alu::execute(self.arch,name,inst,st),
                };
                result.map_err(|e|format!("{symbol} wg={:?} inst={} {name}: {e}{}",launch.workgroup,st.pc.saturating_sub(1),failure_bits(self.arch,wave,st,inst)))?;
            }
            // A gfx12 signal is an arrival, not a wait. A wave may continue
            // executing after its arrival while peers' waits already retire.
            if let Some(arrivals)=states.iter().filter(|s|!s.ended).map(|s|s.signals).min() {
                for st in &mut states {
                    if st.barrier.is_some_and(|phase|phase<arrivals) {st.barrier=None;progress=true;}
                }
            }
            if !progress {return Err("workgroup barrier deadlock".into());}
        }
        Ok(())
    }
}
/// Err-path only: exact architectural bits of the failing wave state and every
/// operand (destinations included, so implicit accumulators are visible).
/// Reads are guarded so diagnostics can never mask the original error.
fn failure_bits(arch:Arch,wave:usize,st:&State,inst:&Inst)->String {
    use std::fmt::Write as _;
    let mut out=String::new();
    let _=write!(out," | arch={arch:?} wave={wave} exec={:#010x} vcc={:#010x} scc={} m0={:#010x} mods={:?} fields={:?}",st.exec,st.vcc,u8::from(st.scc),st.m0,inst.mods,inst.fields);
    match inst.literal {Some(value)=>{let _=write!(out," literal={value:#010x}");},None=>out.push_str(" literal=None")}
    for (i,op) in inst.operands.iter().enumerate() {
        let _=write!(out," | op{i} {op} {op:?}:");
        if let Err(e)=op.validate() {let _=write!(out," invalid({e})");continue;}
        match op {
            Operand::Reg(r)=>for word in 0..usize::from(r.len) {
                if r.kind==Kind::V {
                    let _=write!(out," w{word}=[");
                    for lane in 0..32 {
                        if lane!=0 {out.push(' ');}
                        match st.read(op,lane,word) {Ok(v)=>{let _=write!(out,"{v:#010x}");},Err(e)=>{let _=write!(out,"unreadable({e})");}}
                    }
                    out.push(']');
                } else {
                    match st.read(op,0,word) {Ok(v)=>{let _=write!(out," w{word}={v:#010x}");},Err(e)=>{let _=write!(out," w{word}=unreadable({e})");}}
                }
            },
            Operand::Half(r,_) if r.kind==Kind::V=>{
                out.push_str(" h=[");
                for lane in 0..32 {
                    if lane!=0 {out.push(' ');}
                    match st.read(op,lane,0) {Ok(v)=>{let _=write!(out,"{v:#06x}");},Err(e)=>{let _=write!(out,"unreadable({e})");}}
                }
                out.push(']');
            },
            Operand::Special(Special::Ttmp(n)) if usize::from(*n)>=st.t.len()=>{let _=write!(out," unreadable(ttmp{n} out of range)");},
            _=>match st.read(op,0,0) {Ok(v)=>{let _=write!(out," {v:#010x}");},Err(e)=>{let _=write!(out," unreadable({e})");}},
        }
    }
    out
}
fn control(name:&str,i:&Inst,s:&mut State,k:&Kernel)->Result<()> {
    match name {
        "s_endpgm"=>s.ended=true,
        "s_barrier"=>{s.barrier=Some(s.signals);s.signals+=1;},
        "s_barrier_signal"=>{
            if s.read(&i.operands[0],0,0)?!=u32::MAX {return Err("named barrier signal is not modeled".into());}
            s.signals+=1;
        },
        "s_barrier_wait"=>{
            // SOPP is a 16-bit field: decoded 0xffff is represented as signed -1.
            let id=match &i.operands[0] {
                Operand::Imm(ImmField::Sopp(value))=>*value as u16,
                op=>u16::try_from(s.read(op,0,0)?).map_err(|_|"barrier ID exceeds its 16-bit field")?,
            };
            if id!=u16::MAX {return Err("named barrier wait is not modeled".into());}
            if s.signals==0 {return Err("barrier wait without signal".into());}
            s.barrier=Some(s.signals-1);
        },
        "s_branch"|"s_cbranch_scc0"|"s_cbranch_scc1"|"s_cbranch_execz"|"s_cbranch_execnz"|"s_cbranch_vccz"|"s_cbranch_vccnz"=>{
            let take=match name {"s_branch"=>true,"s_cbranch_scc0"=>!s.scc,"s_cbranch_scc1"=>s.scc,"s_cbranch_execz"=>s.exec==0,"s_cbranch_execnz"=>s.exec!=0,"s_cbranch_vccz"=>s.vcc==0,_=>s.vcc!=0};
            if take {let Operand::Label(id)=i.operands[0] else {return Err("branch missing CFG label".into());};s.pc=k.body.blocks.iter().find(|b|b.id==id).ok_or("branch block missing")?.range.0;}
        },_=>return Err(format!("control opcode {name} unsupported"))
    } Ok(())
}
#[cfg(test)] mod tests {
    use super::*;
    fn state()->State {State{s:[0;106],t:[0;16],v:Box::new([[0;32];256]),exec:u32::MAX,vcc:0,scc:false,m0:0,pc:0,ended:false,barrier:None,signals:0}}
    fn reg(kind:Kind,base:u16,len:u8)->Operand {Operand::Reg(crate::reg::RegRef{kind,base,len})}
    fn v(base:u16)->Operand {reg(Kind::V,base,1)}
    fn s(base:u16)->Operand {reg(Kind::S,base,1)}
    fn insn(name:&str,operands:Vec<Operand>)->Inst {let row=crate::isa::gfx12().iter().find(|r|r.name==name).unwrap();Inst::from_parts(Arch::Gfx1201,row.op,row.form,Default::default(),operands.into_iter().collect(),Default::default(),None,Default::default()).unwrap()}
    fn kernel(body:crate::Body)->Kernel {
        use crate::{descriptor::*,metadata::{HsaKernelMetadata,KernelMeta},inst::{KernelOrigin,SymbolId}};
        Kernel{symbol:SymbolId("exchange".into()),wave:Wave::Wave32,body,
            abi:Abi::Hsa{descriptor:KernelDescriptor{
                group_segment_fixed_size:256,private_segment_fixed_size:0,kernarg_size:0,kernel_code_entry_byte_offset:0,
                compute_pgm_rsrc1:Rsrc1(0xaf0000),compute_pgm_rsrc2:Rsrc2(0x180),compute_pgm_rsrc3:Rsrc3(0),
                kernel_code_properties:KernelCodeProperties(0x400),kernarg_preload:KernargPreload(0),reserved:[0;28]},
                metadata:HsaKernelMetadata{raw_msgpack:vec![],parsed:KernelMeta::default()}},
            origin:KernelOrigin::Authored{builder_crate:"test".into(),version:"1".into(),git:"test".into()}}
    }
    #[test]fn split_barrier_rendezvous_waits_for_delayed_wave() {
        use crate::{cfg::{Block,BlockId,Cond,Terminator},inst::{Setting,Target}};
        let mut body=crate::Body::default();
        let mut add=|name:&str,ops:Vec<Operand>| {let id=body.insts.insert(insn(name,ops));body.layout.push(id);};
        add("s_mov_b32",vec![s(4),Operand::Literal(0x10000)]);
        add("s_mov_b32",vec![s(5),Operand::Literal(0)]);
        add("s_mov_b32",vec![s(6),Operand::Literal(256)]);
        add("s_mov_b32",vec![s(7),Operand::Literal(0x3100_4000)]);
        add("v_lshlrev_b32_e32",vec![v(1),Operand::Inline(InlineConst::Integer(2)),v(0)]);
        add("v_readfirstlane_b32",vec![s(0),v(0)]);
        add("s_cmp_lt_u32",vec![s(0),Operand::Inline(InlineConst::Integer(32))]);
        add("s_cbranch_scc0",vec![Operand::Label(BlockId(3))]);
        add("s_mov_b32",vec![s(8),Operand::Inline(InlineConst::Integer(16))]);
        add("s_sub_co_i32",vec![s(8),s(8),Operand::Inline(InlineConst::Integer(1))]);
        add("s_cmp_lg_u32",vec![s(8),Operand::Inline(InlineConst::Integer(0))]);
        add("s_cbranch_scc1",vec![Operand::Label(BlockId(2))]);
        add("ds_store_b32",vec![v(1),v(0)]);
        add("s_barrier_signal",vec![Operand::Imm(ImmField::Sopp(-1))]);
        add("s_barrier_wait",vec![Operand::Imm(ImmField::Sopp(-1))]);
        add("v_xor_b32_e32",vec![v(2),v(1),Operand::Literal(128)]);
        add("ds_load_b32",vec![v(3),v(2)]);
        add("buffer_store_b32",vec![v(3),v(1),reg(Kind::S,4,4),Operand::Special(Special::Null),Operand::Vmem(crate::operand::VmemToken::Offen)]);
        add("s_endpgm",vec![]);
        for (id,range,term,preds,succs) in [
            (0,(0,8),Terminator::Branch{cond:Cond::Scc0,taken:BlockId(3),fallthrough:BlockId(1)},vec![],vec![3,1]),
            (1,(8,9),Terminator::FallThrough,vec![0],vec![2]),
            (2,(9,12),Terminator::Branch{cond:Cond::Scc1,taken:BlockId(2),fallthrough:BlockId(3)},vec![1,2],vec![2,3]),
            (3,(12,19),Terminator::EndPgm,vec![0,2],vec![]),
        ] {
            body.blocks.push(Block{id:BlockId(id),range,term,preds:preds.into_iter().map(BlockId).collect(),succs:succs.into_iter().map(BlockId).collect()});
        }
        let p=Program{target:Target{arch:Arch::Gfx1201,xnack:Setting::Any,sramecc:Setting::Any,abi_version:4},kernels:vec![kernel(body)],source:None};
        let mut mem=Memory::default();mem.map("out".into(),0x10000,vec![0xcd;256]).unwrap();
        Emulator::new(&p,"exchange").unwrap().run_workgroup(&Launch{kernarg_ptr:0,workgroup:[0,0,0],block:[64,1,1],dynamic_lds:0},&mut mem).unwrap();
        let expected=(0..64u32).flat_map(|n|(n^32).to_le_bytes()).collect::<Vec<_>>();
        assert_eq!(mem.regions[0].bytes,expected);assert_eq!(mem.regions[0].written.count(),256);
    }
    #[test]fn descriptor_selects_packed_ids_and_arch_workgroup_registers() {
        let mut k=kernel(crate::Body::default());
        let Abi::Hsa{descriptor:d,..}=&mut k.abi else{unreachable!()};
        d.compute_pgm_rsrc2.0=0x180|(1<<11);
        let launch=Launch{kernarg_ptr:0,workgroup:[7,9,0],block:[8,8,2],dynamic_lds:0};
        let a=State::new(Arch::Gfx1201,&k,&launch,1).unwrap();
        assert_eq!(a.t[9],7);assert_eq!(a.t[7],9);assert_eq!(a.t[8],(1<<30)|(1<<25));
        assert_eq!(a.v[0][0],4<<10);
        let b=State::new(Arch::Gfx1151,&k,&launch,3).unwrap();
        assert_eq!((b.s[0],b.s[1]),(7,9));assert_eq!(b.v[0][0],4<<10);
        let Abi::Hsa{descriptor:d,..}=&mut k.abi else{unreachable!()};d.compute_pgm_rsrc2.0=0x180|(2<<11);
        let c=State::new(Arch::Gfx1151,&k,&launch,3).unwrap();assert_eq!(c.v[0][0],(1<<20)|(4<<10));
    }
    // RDNA4 ISA scalar unsigned carry differs from signed overflow.
    #[test] fn scalar_carry_propagates_and_mask_updates_scc() {
        let mut st=state();st.s[1]=u32::MAX;st.s[2]=1;
        alu::execute(Arch::Gfx1201,"s_add_co_u32",&insn("s_add_co_u32",vec![s(3),s(1),s(2)]),&mut st).unwrap();
        assert_eq!(st.s[3],0);assert!(st.scc);
        alu::execute(Arch::Gfx1201,"s_add_co_ci_u32",&insn("s_add_co_ci_u32",vec![s(4),s(3),s(3)]),&mut st).unwrap();
        assert_eq!(st.s[4],1);assert!(!st.scc);
        alu::execute(Arch::Gfx1201,"s_and_b32",&insn("s_and_b32",vec![Operand::Special(Special::ExecLo),Operand::Special(Special::ExecLo),s(4)]),&mut st).unwrap();
        assert_eq!(st.exec,1);assert!(st.scc);
    }
    // A raw b64 access straddling NUM_RECORDS bounds is checked per DWORD.
    #[test] fn raw_buffer_zeroes_oob_dword_and_drops_store() {
        let mut st=state();st.exec=1;st.s[4]=0x1000;st.s[6]=4;st.s[7]=0x3100_4000;
        let args=vec![reg(Kind::V,0,2),v(2),reg(Kind::S,4,4),Operand::Special(Special::Null),Operand::Vmem(crate::operand::VmemToken::Offen)];
        let mut mem=Memory::default();mem.map("input".into(),0x1000,0x12345678u32.to_le_bytes().to_vec()).unwrap();
        memory::execute("buffer_load_b64",&insn("buffer_load_b64",args),&mut st,&mut mem,&mut []).unwrap();
        assert_eq!(st.v[0][0],0x12345678);assert_eq!(st.v[1][0],0);
        st.v[2][0]=4;st.v[0][0]=0xdeadbeef;
        let args=vec![v(0),v(2),reg(Kind::S,4,4),Operand::Special(Special::Null),Operand::Vmem(crate::operand::VmemToken::Offen)];
        memory::execute("buffer_store_b32",&insn("buffer_store_b32",args),&mut st,&mut mem,&mut []).unwrap();
        assert_eq!(mem.regions[0].bytes,0x12345678u32.to_le_bytes());assert_eq!(mem.regions[0].written.count(),0);
    }
    // Permlane reads the old source before any aliasing destination writes.
    #[test] fn permlane_swaps_halves_with_inactive_source_policy() {
        let mut st=state();for lane in 0..32{st.v[0][lane]=lane as u32;}
        let i=insn("v_permlanex16_b32",vec![v(0),v(0),Operand::Literal(0x76543210),Operand::Literal(0xfedcba98)]);
        alu::execute(Arch::Gfx1201,"v_permlanex16_b32",&i,&mut st).unwrap();
        assert_eq!(st.v[0][0],16);assert_eq!(st.v[0][16],0);
        st.exec=1;st.v[0][0]=123;let mut i=i;i.mods.op_sel=0;
        alu::execute(Arch::Gfx1201,"v_permlanex16_b32",&i,&mut st).unwrap();assert_eq!(st.v[0][0],123);
        i.mods.op_sel=2;alu::execute(Arch::Gfx1201,"v_permlanex16_b32",&i,&mut st).unwrap();assert_eq!(st.v[0][0],0);
    }
    #[test] fn half_destination_preserves_high_bits_and_exec() {
        let mut st=state();st.exec=1;st.v[0][0]=0xdeadbeef;st.v[0][1]=0xfeedface;st.v[1][0]=1f32.to_bits();
        let dest=Operand::Half(crate::reg::RegRef{kind:Kind::V,base:0,len:1},Half::Lo);
        alu::execute(Arch::Gfx1201,"v_cvt_f16_f32_e32",&insn("v_cvt_f16_f32_e32",vec![dest,v(1)]),&mut st).unwrap();
        assert_eq!(st.v[0][0],0xdead3c00);assert_eq!(st.v[0][1],0xfeedface);
    }
    #[test] fn mapped_access_tracks_identical_stores_and_rejects_crossing() {let mut m=Memory::default();m.map("out".into(),0x1000,vec![1,2,3,4]).unwrap();m.write(0x1001,&[2,9]).unwrap();assert_eq!(m.read(0x1000,4).unwrap(),[1,2,9,4]);assert_eq!((0..4).map(|n|m.regions[0].written.contains(n)).collect::<Vec<_>>(),[false,true,true,false]);assert!(m.read(0x1003,2).is_err());assert!(m.map("alias".into(),0x1002,vec![0;4]).is_err());}
}
