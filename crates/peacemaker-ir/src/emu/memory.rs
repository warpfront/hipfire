use super::{Memory, Result, State};
use crate::{Inst, operand::{Operand, ImmField, VmemToken}};
fn width(name:&str)->Result<usize> {
    if name.contains("b256"){Ok(32)}else if name.contains("b128"){Ok(16)}else if name.contains("b96"){Ok(12)}else if name.contains("b64"){Ok(8)}else if name.contains("b32"){Ok(4)}else if name.contains("b16")||name.contains("u16"){Ok(2)}else if name.contains("u8"){Ok(1)}else{Err(format!("unknown memory width {name}"))}
}
/// Validates the MUBUF operand shape and returns whether OFFEN supplies a vector offset.
/// IDXEN is not representable in the IR: the decoder narrows the VADDR register to one
/// dword for IDXEN xor OFFEN, so a one-dword VADDR without OFFEN, or a wider VADDR
/// with OFFEN (index+offset pair), can only be an index-addressed access and is refused.
fn buffer_shape(ops:&[Operand])->Result<bool> {
    if ops.len()<4 {return Err("buffer instruction requires data, address, SRD and soffset operands".into());}
    for op in &ops[4..] {
        if !matches!(op,Operand::Vmem(VmemToken::Offen)|Operand::Imm(ImmField::VmemOffset(_))|Operand::Scope(_)|Operand::CacheTh(_)) {
            return Err(format!("unsupported buffer operand {op}"));
        }
    }
    let offen=ops[4..].iter().any(|o|matches!(o,Operand::Vmem(VmemToken::Offen)));
    match (&ops[1],offen) {
        (Operand::Reg(r),true) if r.kind==crate::reg::Kind::V&&r.len==1 => Ok(true),
        (Operand::Reg(_),true) => Err("buffer OFFEN requires one 32-bit VGPR; IDXEN addressing unsupported".into()),
        (Operand::Vmem(VmemToken::Off),false) => Ok(false),
        (Operand::Reg(r),false) if r.kind==crate::reg::Kind::V&&r.len>1 => Ok(false),
        (Operand::Reg(_),false) => Err("buffer IDXEN addressing unsupported".into()),
        _ => Err("unsupported buffer address operand".into()),
    }
}
/// RDNA3/RDNA4 V# word3 (descriptor bits 127:96): DST_SEL[11:0] and FORMAT[17:12] are inert for
/// raw accesses (FORMAT==0 is the unbound-resource encoding and is refused), bits 25:24 are compression
/// flags accepted only because captured SRDs (0x31004000) set bit 24; the docs do not state that they are
/// raw-inert. OOB_SELECT is [29:28]. Everything else changes addressing and is refused.
fn raw_srd_word3(word3:u32)->Result<()> {
    if word3>>30!=0 {return Err("buffer SRD TYPE must be 0".into());}
    if word3>>28&3!=3 {return Err("only raw buffer OOB mode 3 is supported".into());}
    // §9.6 Unbound Resources: FORMAT==0 (INVALID) without ADD_TID makes loads return zero and drops stores.
    // That rule is not modeled, so refuse rather than perform a normal memory access.
    if word3>>12&0x3f==0 {return Err("buffer SRD FORMAT 0 is an unbound resource; unmodeled".into());}
    if word3&(1<<23)!=0 {return Err("buffer SRD ADD_TID_ENABLE unsupported".into());}
    if word3&(3<<21)!=0 {return Err("buffer SRD INDEX_STRIDE/swizzle unsupported".into());}
    if word3&0x0c0c_0000!=0 {return Err("buffer SRD stride-scale/compression-access bits unsupported".into());}
    if word3&(1<<20)!=0 {return Err("buffer SRD reserved bit 20 set".into());}
    Ok(())
}
pub(super) fn execute(name:&str,i:&Inst,s:&mut State,mem:&mut Memory,lds:&mut [u8])->Result<()> {
    if i.mods.ds.gds {return Err("GDS not modeled".into());}
    let ops=&i.operands;
    let n=width(name)?;
    if name.starts_with("s_load_") {
        let mut off=0u64;
        for op in &ops[2..] {off=off.wrapping_add(u64::from(s.read(op,0,0)?));}
        let addr=s.read64(&ops[1],0)?.wrapping_add(off);
        for w in 0..n/4 {let bytes=mem.read(addr+4*w as u64,4)?;s.put(&ops[0],0,w,u32::from_le_bytes(bytes.try_into().unwrap()))?;} return Ok(());
    }
    if name=="ds_swizzle_b32" {
        let pat=ops.iter().find_map(|o|if let Operand::Imm(ImmField::DsOffset(n))=o{Some(*n)}else{None}).unwrap_or(0);
        let mut values=[0;32];
        for (lane,val) in values.iter_mut().enumerate() {
            let src=if pat&0x8000!=0 {(lane&!3)|usize::from((pat>>(2*(lane&3)))&3)} else {((lane & usize::from(pat&31))|usize::from((pat>>5)&31)) ^ usize::from((pat>>10)&31)};
            *val=if src<32 && s.exec&(1<<src)!=0 {s.read(&ops[1],src,0)?}else{0};
        }
        for (lane,val) in values.into_iter().enumerate(){if s.exec&(1<<lane)!=0{s.put(&ops[0],lane,0,val)?;}}return Ok(());
    }
    let load=name.contains("load");
    let ds=name.starts_with("ds_");
    let buffer=name.starts_with("buffer_");
    let two=name.contains("2addr");
    for lane in 0..32 {
        if s.exec&(1<<lane)==0 {continue;}
        let mut offset=0i64;let mut off0=0usize;let mut off1=0usize;
        for op in ops {match op {Operand::Imm(ImmField::VmemOffset(n))=>offset+=i64::from(*n),Operand::Imm(ImmField::DsOffset(n))=>offset+=i64::from(*n),Operand::Imm(ImmField::DsOffset0(n))=>off0=usize::from(*n),Operand::Imm(ImmField::DsOffset1(n))=>off1=usize::from(*n),_=>()}}
        let mut addr;let mut raw_bounds=None;
        if ds {addr=u64::from(s.read(&ops[usize::from(load)],lane,0)?).wrapping_add_signed(offset);}
        else if buffer {
            let offen=buffer_shape(ops)?;
            let Operand::Reg(srd)=ops[2] else{return Err("buffer SRD missing".into());};
            if srd.kind!=crate::reg::Kind::S||srd.len!=4{return Err("buffer SRD requires four scalar registers".into());}
            let r=usize::from(srd.base);let word1=s.s[r+1];let word3=s.s[r+3];
            if word1>>16 != 0 {return Err("strided or swizzled buffer descriptor unsupported".into());}
            raw_srd_word3(word3)?;
            // Raw byte-addressed SRDs: base[47:0], NUM_RECORDS in bytes.
            let base=u64::from(s.s[r])|(u64::from(word1&0xffff)<<32);
            let vaddr=if offen {u64::from(s.read(&ops[1],lane,0)?)}else{0};
            let soff=u64::from(s.read(&ops[3],lane,0)?);
            let relative=vaddr.wrapping_add(soff).wrapping_add_signed(offset);
            raw_bounds=Some((relative,u64::from(s.s[r+2])));
            addr=base.wrapping_add(relative);
        } else if name.starts_with("global_") || name.starts_with("flat_") {
            let a=if load {1}else{0};addr=s.read64(&ops[a],lane)?.wrapping_add_signed(offset);
            if let Some(Operand::Reg(r))=ops.get(if load{2}else{2}) {if r.kind==crate::reg::Kind::S {addr=addr.wrapping_add(s.read64(&ops[2],lane)?);}}
        } else {return Err(format!("memory opcode {name} unsupported"));}
        let data=if load{&ops[0]}else if buffer{&ops[0]}else{&ops[1]};
        let count=if two{2}else{1};
        for part in 0..count {
            let cur=if two{addr+if part==0{off0}else{off1} as u64*n as u64}else{addr};
            let data_word=part*n.div_ceil(4);
            if load {
                let mut bytes=[0u8;16];
                for offset in (0..n).step_by(4) {
                    let size=(n-offset).min(4);
                    if raw_bounds.is_some_and(|(relative,limit)|relative.checked_add((offset+size)as u64).is_none_or(|end|end>limit)) {continue;}
                    let src=if ds {lds.get(cur as usize+offset..cur as usize+offset+size).ok_or_else(||format!("LDS read out of bounds {cur:#x}+{n}"))?}else{mem.read(cur+offset as u64,size)?};
                    bytes[offset..offset+size].copy_from_slice(src);
                }
                for w in 0..n.div_ceil(4) {let mut val=u32::from_le_bytes(bytes[w*4..w*4+4].try_into().unwrap());
                    if name.contains("d16_hi") {val=(s.read(data,lane,data_word+w)?&0xffff)|(val<<16);}
                    else if name.contains("d16") {val=(s.read(data,lane,data_word+w)?&0xffff0000)|(val&0xffff);}
                    s.put(data,lane,data_word+w,val)?;
                }
            } else {
                let mut bytes=[0u8;16];for w in 0..n.div_ceil(4) {let mut val=s.read(data,lane,data_word+w)?;if name.contains("d16_hi"){val>>=16;}bytes[w*4..w*4+4].copy_from_slice(&val.to_le_bytes());}
                for offset in (0..n).step_by(4) {
                    let size=(n-offset).min(4);
                    if raw_bounds.is_some_and(|(relative,limit)|relative.checked_add((offset+size)as u64).is_none_or(|end|end>limit)) {continue;}
                    if ds {lds.get_mut(cur as usize+offset..cur as usize+offset+size).ok_or_else(||format!("LDS write out of bounds {cur:#x}+{n}"))?.copy_from_slice(&bytes[offset..offset+size]);}else{mem.write(cur+offset as u64,&bytes[offset..offset+size])?;}
                }
            }
        }
    }Ok(())
}
#[cfg(test)] mod tests {
    use super::*;
    use crate::{Arch,reg::{Kind,RegRef}};
    fn state()->State {State{s:[0;106],t:[0;16],v:Box::new([[0;32];256]),exec:1,vcc:0,scc:false,m0:0,pc:0,ended:false,barrier:None,signals:0}}
    fn reg(kind:Kind,base:u16,len:u8)->Operand {Operand::Reg(RegRef{kind,base,len})}
    fn insn(name:&str,operands:Vec<Operand>)->Inst {let row=crate::isa::gfx12().iter().find(|r|r.name==name).unwrap();Inst::from_parts(Arch::Gfx1201,row.op,row.form,Default::default(),operands.into_iter().collect(),Default::default(),None,Default::default()).unwrap()}
    fn setup(word3:u32)->(State,Memory) {
        let mut st=state();st.s[4]=0x1000;st.s[6]=8;st.s[7]=word3;st.v[2][0]=4;
        let mut mem=Memory::default();mem.map("buf".into(),0x1000,(0..8u8).collect()).unwrap();(st,mem)
    }
    fn load(st:&mut State,mem:&mut Memory,vaddr:Operand,extra:Vec<Operand>)->Result<()> {
        let mut ops=vec![reg(Kind::V,0,1),vaddr,reg(Kind::S,4,4),Operand::Special(crate::operand::Special::Null)];ops.extend(extra);
        execute("buffer_load_b32",&insn("buffer_load_b32",ops),st,mem,&mut [])
    }
    fn offen()->Operand {Operand::Vmem(VmemToken::Offen)}
    // Captured raw SRDs (0x31004000) and offset-only/offen forms keep exact byte-bounds semantics.
    #[test] fn raw_forms_and_captured_srd_word3_are_accepted() {
        let (mut st,mut mem)=setup(0x3100_4000);
        load(&mut st,&mut mem,reg(Kind::V,2,1),vec![offen()]).unwrap();
        assert_eq!(st.v[0][0],u32::from_le_bytes([4,5,6,7]));
        load(&mut st,&mut mem,Operand::Vmem(VmemToken::Off),vec![]).unwrap();
        assert_eq!(st.v[0][0],u32::from_le_bytes([0,1,2,3]));
        load(&mut st,&mut mem,reg(Kind::V,2,2),vec![]).unwrap();
        assert_eq!(st.v[0][0],u32::from_le_bytes([0,1,2,3]));
        st.v[2][0]=8; // offset 8 == NUM_RECORDS: out of bounds reads zero
        load(&mut st,&mut mem,reg(Kind::V,2,1),vec![offen()]).unwrap();
        assert_eq!(st.v[0][0],0);
    }
    #[test] fn idxen_shaped_addresses_are_hard_errors() {
        for (vaddr,extra) in [(reg(Kind::V,2,1),vec![]),(reg(Kind::V,2,2),vec![offen()])] {
            let (mut st,mut mem)=setup(0x3100_4000);st.v[0][0]=0xdead_beef;
            let e=load(&mut st,&mut mem,vaddr,extra).unwrap_err();
            assert!(e.contains("IDXEN"),"{e}");assert_eq!(st.v[0][0],0xdead_beef);
        }
    }
    #[test] fn unsupported_srd_word3_and_extra_operands_are_hard_errors() {
        for (word3,needle) in [
            (0x3100_4000|1<<30,"TYPE"),(0x3100_4000|1<<31,"TYPE"),(0x3100_4000|1<<23,"ADD_TID"),
            (0x3100_4000|1<<21,"INDEX_STRIDE"),(0x3100_4000|3<<21,"INDEX_STRIDE"),
            (0x3100_4000|1<<18,"stride-scale"),(0x3100_4000|1<<26,"compression-access"),(0x3100_4000|1<<20,"reserved"),
            (0x2100_4000,"OOB"),(0x3000_0000,"FORMAT 0"),(0x3100_0fff,"FORMAT 0"),
        ] {
            let (mut st,mut mem)=setup(word3);st.v[0][0]=0xdead_beef;
            let e=load(&mut st,&mut mem,reg(Kind::V,2,1),vec![offen()]).unwrap_err();
            assert!(e.contains(needle),"{word3:#x}: {e}");assert_eq!(st.v[0][0],0xdead_beef);
        }
        let (mut st,mut mem)=setup(0x3100_4000);
        st.s[5]=1<<16; // SRD stride field
        assert!(load(&mut st,&mut mem,reg(Kind::V,2,1),vec![offen()]).is_err());
        let (mut st,mut mem)=setup(0x3100_4000);
        let e=load(&mut st,&mut mem,reg(Kind::V,2,1),vec![offen(),Operand::Literal(0)]).unwrap_err();
        assert!(e.contains("unsupported buffer operand"),"{e}");
    }
}
