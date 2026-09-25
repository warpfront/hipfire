use super::{Spec, END, so, vo, op, mem, s, sr, v, publish};
use crate::{Builder, insn::MemoryClass, lds::Transition};

// Exact unsigned SALU division: reciprocal estimate, high product and two
// quotient corrections (the same verified expansion as iu4_gemm::prologue).
fn udiv(b:&mut Builder,num:u8,den:u8,q:u8,r:u8,x:u8,y:u8)->Result<(),String>{
    so(b,format!("s_cvt_f32_u32 s{x}, s{den}"),&[x],&[den])?;
    so(b,format!("v_s_rcp_f32 s{x}, s{x}"),&[x],&[x])?;
    so(b,format!("s_mul_f32 s{x}, s{x}, 0x4f7ffffe"),&[x],&[x])?;
    so(b,format!("s_cvt_u32_f32 s{x}, s{x}"),&[x],&[x])?;
    so(b,format!("s_sub_co_i32 s{y}, 0, s{den}"),&[y],&[den])?;
    so(b,format!("s_mul_i32 s{y}, s{y}, s{x}"),&[y],&[y,x])?;
    so(b,format!("s_mul_hi_u32 s{y}, s{x}, s{y}"),&[y],&[x,y])?;
    so(b,format!("s_add_co_i32 s{x}, s{x}, s{y}"),&[x],&[x,y])?;
    so(b,format!("s_mul_hi_u32 s{q}, s{num}, s{x}"),&[q],&[num,x])?;
    so(b,format!("s_mul_i32 s{y}, s{q}, s{den}"),&[y],&[q,den])?;
    so(b,format!("s_sub_co_i32 s{r}, s{num}, s{y}"),&[r],&[num,y])?;
    for _ in 0..2 {
        so(b,format!("s_add_co_i32 s{x}, s{q}, 1"),&[x],&[q])?;
        so(b,format!("s_sub_co_i32 s{y}, s{r}, s{den}"),&[y],&[r,den])?;
        so(b,format!("s_cmp_ge_u32 s{r}, s{den}"),&[],&[r,den])?;
        so(b,format!("s_cselect_b32 s{q}, s{x}, s{q}"),&[q],&[x,q])?;
        so(b,format!("s_cselect_b32 s{r}, s{y}, s{r}"),&[r],&[y,r])?;
    }
    Ok(())
}
fn descriptor(b:&mut Builder,dst:u8,src:u8,records:Option<u8>)->Result<(),String>{
    so(b,format!("s_mov_b32 s{dst}, s{src}"),&[dst],&[src])?;
    so(b,format!("s_and_b32 s{}, s{}, 0xffff",dst+1,src+1),&[dst+1],&[src+1])?;
    if let Some(n)=records {so(b,format!("s_mov_b32 s{}, s{n}",dst+2),&[dst+2],&[n])?;}
    else {so(b,format!("s_mov_b32 s{}, -1",dst+2),&[dst+2],&[])?;}
    so(b,format!("s_mov_b32 s{}, 0x31004000",dst+3),&[dst+3],&[])
}

pub(super) fn emit(b:&mut Builder,spec:Spec)->Result<(),String>{
    for i in 0..3 {let a=8+i*8;let off=i*32;
        mem(b,format!("s_load_b256 s[{a}:{}], s[0:1], {off:#x}",a+7),&[sr(a,8)],&[sr(0,2)],MemoryClass::SmemLoad)?;
    }
    // Mt and ceil(Mt/256), ceil(N/128), both computed once per CTA.
    so(b,"s_add_co_i32 s80, s26, s27",&[80],&[26,27])?;
    if spec.epi != super::Epi::GateUpSilu {
        for n in 28..26+spec.epi.count() as u8 {so(b,format!("s_add_co_i32 s80, s80, s{n}"),&[80],&[80,n])?;}
    }
    so(b,"s_add_co_i32 s80, s80, 0xff",&[80],&[80])?;
    so(b,"s_lshr_b32 s80, s80, 8",&[80],&[80])?;
    so(b,"s_add_co_i32 s81, s31, 0x7f",&[81],&[31])?;
    so(b,"s_lshr_b32 s81, s81, 7",&[81],&[81])?;
    so(b,"s_and_b32 s95, ttmp7, 0xffff",&[95],&[])?;
    so(b,"s_mul_i32 s95, s95, s80",&[95],&[95,80])?;
    so(b,"s_add_co_i32 s95, s95, ttmp9",&[95],&[95])?;
    so(b,"s_lshl_b32 s96, s80, 3",&[96],&[80])?;
    udiv(b,95,96,97,98,99,100)?; // band, q
    so(b,"s_lshl_b32 s99, s97, 3",&[99],&[97])?;
    so(b,"s_sub_co_i32 s100, s81, s99",&[100],&[81,99])?;
    so(b,"s_min_u32 s100, s100, 8",&[100],&[100])?;
    udiv(b,98,100,101,102,103,72)?; // row, tile in band
    so(b,"s_mov_b32 s83, s101",&[83],&[101])?;
    so(b,"s_lshl_b32 s82, s97, 3",&[82],&[97])?;
    so(b,"s_add_co_i32 s82, s82, s102",&[82],&[82,102])?;
    so(b,"s_lshl_b32 s82, s82, 7",&[82],&[82])?;
    so(b,"s_lshr_b32 s84, s30, 7",&[84],&[30])?;
    so(b,"s_mov_b32 s85, 0",&[85],&[])?;
    // The launch grid itself guarantees valid tiles. This guard still makes
    // accidental oversubscription harmless without indexing outside the ABI.
    so(b,"s_cmp_ge_u32 s82, s31",&[],&[82,31])?;
    op(b,format!("s_cbranch_scc1 {END}"),&[],&[])?;
    so(b,"s_mov_b32 s94, s83",&[94],&[83])?;
    so(b,"s_mul_i32 s94, s94, s84",&[94],&[94,84])?;
    op(b,"v_readfirstlane_b32 s87, v0",&[s(87)],&[v(0)])?;
    so(b,"s_lshr_b32 s87, s87, 5",&[87],&[87])?;
    so(b,"s_and_b32 s88, s87, 3",&[88],&[87])?;
    so(b,"s_lshr_b32 s89, s87, 2",&[89],&[87])?;
    // SRDs: raw Wf, X8, Rw, Ew, D and up to four independent output bases.
    so(b,"s_mul_i32 s73, s31, s30",&[73],&[31,30])?;
    so(b,"s_lshr_b32 s74, s73, 5",&[74],&[73])?;
    descriptor(b,32,8,None)?;descriptor(b,36,14,Some(73))?;
    descriptor(b,40,10,None)?;descriptor(b,44,12,None)?;descriptor(b,48,16,Some(74))?;
    for i in 0..4 {descriptor(b,52+i*4,18+i*2,None)?;}
    // Lane address map; v0 is the workgroup-local thread ID in this prologue.
    vo(b,"v_mov_b32_e32 v187, v0",&[187],&[0],&[])?;
    vo(b,"v_and_b32_e32 v176, 31, v0",&[176],&[0],&[])?;
    vo(b,"v_lshlrev_b32_e32 v176, 4, v176",&[176],&[176],&[])?;
    vo(b,"v_lshrrev_b32_e32 v177, 1, v0",&[177],&[0],&[])?;
    vo(b,"v_mul_lo_u32 v177, v177, s30",&[177],&[177],&[30])?;
    vo(b,"v_and_b32_e32 v170, 1, v0",&[170],&[0],&[])?;
    vo(b,"v_lshlrev_b32_e32 v170, 5, v170",&[170],&[170],&[])?;
    vo(b,"v_add_nc_u32_e32 v177, v177, v170",&[177],&[177,170],&[])?;
    so(b,"s_mul_i32 s72, s82, s30",&[72],&[82,30])?;
    vo(b,"v_add_nc_u32_e32 v177, s72, v177",&[177],&[177],&[72])?;
    vo(b,"v_lshrrev_b32_e32 v188, 1, v0",&[188],&[0],&[])?;
    vo(b,"v_lshrrev_b32_e32 v178, 4, v188",&[178],&[188],&[])?;
    vo(b,"v_lshlrev_b32_e32 v178, 8, v178",&[178],&[178],&[])?;
    vo(b,"v_and_b32_e32 v171, 15, v188",&[171],&[188],&[])?;
    vo(b,"v_lshlrev_b32_e32 v171, 3, v171",&[171],&[171],&[])?;
    vo(b,"v_add_nc_u32_e32 v178, v178, v171",&[178],&[178,171],&[])?;
    vo(b,"v_and_b32_e32 v171, 1, v0",&[171],&[0],&[])?;
    vo(b,"v_lshlrev_b32_e32 v171, 12, v171",&[171],&[171],&[])?;
    vo(b,"v_add_nc_u32_e32 v178, v178, v171",&[178],&[178,171],&[])?;
    vo(b,"v_and_b32_e32 v179, 31, v0",&[179],&[0],&[])?;
    vo(b,"v_lshlrev_b32_e32 v179, 3, v179",&[179],&[179],&[])?;
    so(b,"s_lshl_b32 s72, s89, 10",&[72],&[89])?;
    vo(b,"v_add_nc_u32_e32 v179, s72, v179",&[179],&[179],&[72])?;
    vo(b,"v_lshlrev_b32_e32 v182, 2, v0",&[182],&[0],&[])?;
    vo(b,"v_and_b32_e32 v183, 0x7f, v0",&[183],&[0],&[])?;
    vo(b,"v_add_nc_u32_e32 v183, s82, v183",&[183],&[183],&[82])?;
    vo(b,"v_mul_lo_u32 v183, v183, s84",&[183],&[183],&[84])?;
    vo(b,"v_lshlrev_b32_e32 v183, 2, v183",&[183],&[183],&[])?;
    vo(b,"v_and_b32_e32 v180, 31, v0",&[180],&[0],&[])?;
    vo(b,"v_lshrrev_b32_e32 v180, 4, v180",&[180],&[180],&[])?;
    vo(b,"v_lshlrev_b32_e32 v180, 5, v180",&[180],&[180],&[])?;
    so(b,"s_lshl_b32 s72, s88, 8",&[72],&[88])?;
    vo(b,"v_add_nc_u32_e32 v180, s72, v180",&[180],&[180],&[72])?;
    vo(b,"v_and_b32_e32 v181, 15, v0",&[181],&[0],&[])?;
    vo(b,"v_lshlrev_b32_e32 v181, 2, v181",&[181],&[181],&[])?;
    so(b,"s_lshl_b32 s72, s89, 8",&[72],&[89])?;
    vo(b,"v_add_nc_u32_e32 v181, s72, v181",&[181],&[181],&[72])?;
    so(b,"s_lshl_b32 s95, s88, 13",&[95],&[88])?;
    // First slab A0: all 8192 bytes are published once, including masked
    // token lanes, whose raw-buffer out-of-range reads return zero.
    publish::stage_a(b,0,0)?;
    b.barrier(&[Transition::Ready(0)])?;
    Ok(())
}
